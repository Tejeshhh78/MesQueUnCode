"""Usage: python -m possession --team MesQueUnCode --output-dir artifacts"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import sys
import time

import numpy as np
import pandas as pd

from games import GAME, GAMES, half_video_blocks, require
from evaluation import possession_metrics as pm
from . import __version__
from .data import read_events, read_tracking
from .model import PROFILES, predict


ROOT = Path(__file__).resolve().parents[1]


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


def run(data_dir, output_dir, team, profile="v1"):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", team):
        raise ValueError("Team name: 1–80 letters/digits, hyphens or underscores")
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; choose one of {sorted(PROFILES)}")
    started = time.perf_counter()
    config = PROFILES[profile]
    game = GAMES[GAME]
    paths = require(["atd_meta", "atd_raw", "smart_tagging"], data_dir=data_dir)
    fingerprints = {Path(p).name: sha256(p) for p in paths}
    print("Reading and validating tactical tracking locally ...", flush=True)
    features = read_tracking(paths[0], paths[1], game["n_video_frames"], game["pitch"])
    events = read_events(paths[2], game["n_video_frames"], game["fps"])
    print("Computing evidence and temporal states ...", flush=True)
    labels, model_report = predict(features, events, half_video_blocks(GAME), game["pitch"], game["fps"], config)
    output_dir = Path(output_dir)
    target = output_dir / f"{team}.csv"
    write_submission(target, labels)
    codes = pm.label_codes(labels)
    runs = pm.valid_runs(codes)
    spells = pm.possession_spells(codes, runs)
    source_paths = sorted((ROOT / "possession").glob("*.py")) + [ROOT / "games.py", ROOT / "requirements-solution.txt"]
    manifest = {
        "version": __version__, "team": team, "game": GAME, "profile": profile,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "config": config.to_dict(), "inputs_sha256": fingerprints,
        "code_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in source_paths},
        "submission_sha256": sha256(target), "rows": len(labels),
        "label_counts": dict(sorted(Counter(labels.tolist()).items())),
        "possession_count": len(spells), "model": model_report,
        "ground_truth_used": False,
    }
    with (output_dir / f"{team}.manifest.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    document = ROOT / "TECHNICAL_REPORT.md"
    if document.is_file():
        shutil.copyfile(document, output_dir / f"{team}.md")
    print(f"OK: {len(labels):,} rows; {len(spells)} predicted possession spells")
    print(f"SHA-256: {manifest['submission_sha256']}")
    print(f"Elapsed: {time.perf_counter() - started:.2f} s; output: {target}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Local possession inference without human training labels")
    parser.add_argument("--data-dir", help="Data bundle root; default: data/ or MTAH_DATA_DIR")
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts"))
    parser.add_argument("--team", default="MesQueUnCode", help="Registered team name to use as the filename")
    parser.add_argument("--profile", default="v1", choices=sorted(PROFILES),
                        help="v1: method fixed before the public check (default); "
                             "v2: recalibrated on other matches only, see validation/")
    args = parser.parse_args()
    try:
        run(args.data_dir, args.output_dir, args.team, args.profile)
    except (ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
