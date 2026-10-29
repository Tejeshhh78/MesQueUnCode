"""Usage: python -m solution.code.possession --team MesQueUnCode"""

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np
import pandas as pd

from games import GAME, GAMES, half_video_blocks, require
from evaluation import possession_metrics as pm
from . import __version__
from .data import read_events, read_tracking
from .model import PROFILES, predict, profile_config


ROOT = Path(__file__).resolve().parents[3]
SOLUTION_ROOT = ROOT / "solution"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_submission(path, labels):
    """Write platform-independent UTF-8/LF output; replace atomically after validation."""
    path = Path(path)
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.isin(labels, ["H", "A", "D", "X"]).all():
        raise ValueError("Invalid output labels")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("frame,label\n")
            handle.writelines(f"{frame},{label}\n" for frame, label in enumerate(labels))
        reason = pm.validate_submission(pm.read_label_csv(temporary), len(labels))
        if reason:
            raise ValueError(f"Official format validation: {reason}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run(data_dir, output_dir, team, *, profile="incumbent", diagnostics=False, save_trace=False,
        team_switch_cost=None):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", team):
        raise ValueError("Team name: 1–80 letters/digits, hyphens or underscores")
    if profile not in PROFILES:
        raise ValueError(f"Unknown model profile: {profile}")
    if save_trace and not diagnostics:
        raise ValueError("Saving a detailed trace requires diagnostics")
    config = profile_config(profile)
    if team_switch_cost is not None:
        if profile != "responsive" or not np.isfinite(team_switch_cost) or team_switch_cost <= 0:
            raise ValueError("A positive finite team-switch cost requires the responsive profile")
        config = replace(config, team_switch_cost=float(team_switch_cost))
    output_dir = Path(output_dir)
    target = output_dir / f"{team}.csv"
    manifest_path = output_dir / f"{team}.manifest.json"
    if target.exists():
        if not manifest_path.is_file():
            raise ValueError("An output CSV exists without a provenance manifest; choose another output directory")
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("profile", "incumbent") != profile:
            raise ValueError("Output belongs to a different model profile; choose another output directory")
        if prior.get("config") != config.to_dict():
            raise ValueError("Output belongs to a different or undocumented configuration; choose another output directory")
    started = time.perf_counter()
    game = GAMES[GAME]
    paths = require(["atd_meta", "atd_raw", "smart_tagging"], data_dir=data_dir)
    fingerprints = {Path(p).name: sha256(p) for p in paths}
    print("Reading and validating tactical tracking locally ...", flush=True)
    features = read_tracking(paths[0], paths[1], game["n_video_frames"], game["pitch"])
    events = read_events(paths[2], game["n_video_frames"], game["fps"])
    print("Computing evidence and temporal states ...", flush=True)
    blocks = half_video_blocks(GAME)
    trace = {} if diagnostics else None
    labels, model_report = predict(features, events, blocks, game["pitch"], game["fps"], config,
                                   profile=profile, trace=trace)
    diagnostic_paths = None
    if diagnostics:
        from .diagnostics import write_diagnostics
        diagnostic_paths = write_diagnostics(output_dir / f"{team}.diagnostics", trace, labels,
                                             blocks, game["fps"], include_trace=save_trace)
    write_submission(target, labels)
    codes = pm.label_codes(labels)
    runs = pm.valid_runs(codes)
    spells = pm.possession_spells(codes, runs)
    source_paths = sorted(Path(__file__).resolve().parent.glob("*.py")) + [
        ROOT / "games.py", SOLUTION_ROOT / "requirements.txt"]
    manifest = {
        "version": __version__, "team": team, "game": GAME, "profile": profile,
        "status": "incumbent" if profile == "incumbent" else "experimental_not_promoted",
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "config": config.to_dict(), "inputs_sha256": fingerprints,
        "code_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in source_paths},
        "submission_sha256": sha256(target), "rows": len(labels),
        "label_counts": dict(sorted(Counter(labels.tolist()).items())),
        "possession_count": len(spells), "model": model_report,
        "ground_truth_used": False, "diagnostics": diagnostic_paths,
    }
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    document = SOLUTION_ROOT / "TECHNICAL_REPORT.md"
    if document.is_file():
        report_text = document.read_text(encoding="utf-8")
        if profile != "incumbent":
            report_text = (f"> EXPERIMENTAL OUTPUT: profile `{profile}` has not been promoted.\n"
                           "> The incumbent described below remains the submission default.\n\n"
                           + report_text)
        (output_dir / f"{team}.md").write_text(report_text, encoding="utf-8", newline="\n")
    print(f"OK: {len(labels):,} rows; {len(spells)} predicted possession spells")
    print(f"SHA-256: {manifest['submission_sha256']}")
    print(f"Elapsed: {time.perf_counter() - started:.2f} s; output: {target}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Local possession inference without human training labels")
    parser.add_argument("--data-dir", help="Data bundle root; default: data/ or MTAH_DATA_DIR")
    parser.add_argument("--output-dir", help="Default: solution/results for incumbent; separate reports directory for experiments")
    parser.add_argument("--team", default="MesQueUnCode", help="Registered team name to use as the filename")
    parser.add_argument("--profile", choices=PROFILES, default="incumbent",
                        help="Explicit experimental alternative; incumbent remains the default")
    parser.add_argument("--team-switch-cost", type=float,
                        help="Explicit responsive-profile configuration; recorded in the output manifest")
    parser.add_argument("--diagnostics", action="store_true", help="Write aggregate evidence diagnostics")
    parser.add_argument("--save-trace", action="store_true", help="Also save local arrays; requires --diagnostics and a Git-ignored output path")
    args = parser.parse_args()
    if args.save_trace and not args.diagnostics:
        parser.error("--save-trace requires --diagnostics")
    output = args.output_dir or str(SOLUTION_ROOT / "results" if args.profile == "incumbent"
                                    else SOLUTION_ROOT / "reports" / "experimental" / args.profile)
    if args.profile != "incumbent":
        print("Explicit candidate profile: consult independent validation for its evidence and limits. "
              "Incumbent remains the default.", flush=True)
    try:
        run(args.data_dir, output, args.team, profile=args.profile,
            diagnostics=args.diagnostics, save_trace=args.save_trace, team_switch_cost=args.team_switch_cost)
    except (ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
