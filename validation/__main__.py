"""Usage: python -m validation [--download] [--profiles v1 v2] [--seeds 0 1] [--quick]

Scores possession profiles and the starter baseline on Metrica's public sample games (OTHER
matches), after degrading their clean tracking towards the challenge's automatic ATD, with the
official metric code. Reads nothing of the provided match. Results: artifacts/validation/.
"""

import argparse
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

from evaluation import possession_metrics as pm
from possession.model import PROFILES

from . import harness, metrica
from .degrade import Degradation
from .tags import TagModel, synthesise

ROOT = Path(__file__).resolve().parents[1]

# Assumptions about the automatic tracker and the tagger. Conclusions should hold across all.
REGIMES = {
    "default": Degradation(),
    "noisy": Degradation(ball_noise_m=0.6, player_noise_m=0.4),
    "very_noisy": Degradation(ball_noise_m=1.0, player_noise_m=0.5, outlier_rate=0.005),
    "sparse": Degradation(alive_missing=0.45, mean_gap_s=2.0),
}
TAG_MODELS = {
    "none": None,
    "good": TagModel(wrong_team=0.05),
    "noisy": TagModel(wrong_team=0.25, end_jitter_s=3.0),
}
METRICS = ("spellF1", "macroF1", "f1@0.5", "n_pred", "n_gt", "D_f1", "team_acc")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--download", action="store_true", help="fetch the public sample games first")
    parser.add_argument("--data-dir", help="folder holding metrica_sample_game_*/ (default data/external)")
    parser.add_argument("--games", type=int, nargs="+", default=[1, 2, 3], choices=[1, 2, 3])
    parser.add_argument("--profiles", nargs="+", default=sorted(PROFILES), choices=sorted(PROFILES))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--regimes", nargs="+", default=list(REGIMES), choices=list(REGIMES))
    parser.add_argument("--tags", nargs="+", default=list(TAG_MODELS), choices=list(TAG_MODELS))
    parser.add_argument("--quick", action="store_true", help="one seed, default regime, no tags")
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "validation"))
    args = parser.parse_args()
    if args.quick:
        args.seeds, args.regimes, args.tags = [0], ["default"], ["none"]
    if args.download:
        metrica.download(args.games, args.data_dir)
    started = time.perf_counter()
    rows = []
    for regime in args.regimes:
        for game_id, seed, features, gt, blocks, game in harness.prepared(
                args.games, args.seeds, REGIMES[regime], args.data_dir):
            runs = pm.valid_runs(gt)
            for tag_name in args.tags:
                model = TAG_MODELS[tag_name]
                events = [] if model is None else synthesise(gt, runs, game.fps, model, seed=game_id * 100 + seed)
                candidates = {"baseline_smooth50": harness.baseline_labels(features)} if tag_name == "none" else {}
                for profile in args.profiles:
                    candidates[profile] = harness.model_labels(features, blocks, game.pitch, game.fps,
                                                               PROFILES[profile], events)
                for name, labels in candidates.items():
                    result = harness.score(gt, labels)
                    rows.append({"regime": regime, "tags": tag_name, "game": game_id, "seed": seed,
                                 "method": name, **{k: result[k] for k in METRICS}})
            print(f"  {regime:10s} game {game_id} seed {seed} done ({time.perf_counter() - started:.0f} s)",
                  flush=True)
    summary = summarise(rows, args.profiles)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {"python": platform.python_version(), "games": args.games, "seeds": args.seeds,
              "regimes": {k: REGIMES[k].__dict__ for k in args.regimes},
              "tag_models": {k: (TAG_MODELS[k].__dict__ if TAG_MODELS[k] else None) for k in args.tags},
              "profiles": {k: PROFILES[k].to_dict() for k in args.profiles},
              "provided_match_data_used": False, "rows": rows, "summary": summary}
    target = output / ("results-quick.json" if args.quick else "results.json")
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {target}")
    return 0


def summarise(rows, profiles):
    print("\nmean over games x seeds; paired difference vs v1 with its min..max over games x seeds")
    print(f"{'regime':10s} {'tags':6s} {'method':18s} {'spellF1':>8s} {'macroF1':>8s} {'pred/gt':>8s}"
          f" {'D F1':>6s} {'d spellF1 vs v1':>22s}")
    summary = []
    keys = sorted({(r["regime"], r["tags"]) for r in rows}, key=lambda k: (list(REGIMES).index(k[0]), k[1]))
    for regime, tags in keys:
        subset = [r for r in rows if r["regime"] == regime and r["tags"] == tags]
        reference = {(r["game"], r["seed"]): r["spellF1"] for r in subset if r["method"] == "v1"}
        for method in ["baseline_smooth50"] + list(profiles):
            chosen = [r for r in subset if r["method"] == method]
            if not chosen:
                continue
            spell = np.array([r["spellF1"] for r in chosen])
            entry = {"regime": regime, "tags": tags, "method": method, "n": len(chosen),
                     "spellF1": float(spell.mean()), "spellF1_std": float(spell.std()),
                     "macroF1": float(np.mean([r["macroF1"] for r in chosen])),
                     "pred_over_gt": float(np.mean([r["n_pred"] / r["n_gt"] for r in chosen])),
                     "D_f1": float(np.mean([r["D_f1"] for r in chosen]))}
            delta = ""
            if reference and method != "v1":
                diffs = np.array([r["spellF1"] - reference[(r["game"], r["seed"])] for r in chosen])
                entry.update(delta_mean=float(diffs.mean()), delta_min=float(diffs.min()),
                             delta_max=float(diffs.max()))
                delta = f"{diffs.mean():+.4f} ({diffs.min():+.4f}..{diffs.max():+.4f})"
            summary.append(entry)
            print(f"{regime:10s} {tags:6s} {method:18s} {entry['spellF1']:8.4f} {entry['macroF1']:8.4f}"
                  f" {entry['pred_over_gt']:8.2f} {entry['D_f1']:6.3f} {delta:>22s}")
    return summary


if __name__ == "__main__":
    sys.exit(main())
