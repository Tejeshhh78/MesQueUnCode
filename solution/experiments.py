"""Compare the reference baselines, the solution and its ablations in one table.

    python solution/experiments.py
    python solution/experiments.py --write-csv          # also writes every variant's CSV
    python solution/experiments.py --report solution/reports/experiments.md

Two kinds of numbers, kept in separate columns on purpose:

  * FULL MATCH, no labels: possession count, mean possession length and the H/A/D shares over all
    158,857 graded frames. These describe what each system *claims*; they need no ground truth and
    are the only full-match numbers anyone outside the organisers can compute.
  * PUBLIC 5 MINUTES: macroF1 and spellF1 against evaluation/gt/*_public.csv, through the official
    possession_metrics.evaluate(). This is a pipeline check over 17 possessions. EVALUATION.md
    shows it ranks even the four baselines in the wrong order, so it is printed, never optimised:
    no parameter in solution/config.py was chosen by looking at it.

The baselines are rebuilt from the cached tracking with the starter's own smoothing function, and
checked against evaluation/results/baseline_*.csv when those files exist.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from games import GAME, GAMES  # noqa: E402
from solution import config, features, loaders, postprocess  # noqa: E402
from solution.run_solution import predict  # noqa: E402
from starter.baseline import VARIANTS, smooth  # noqa: E402

BASELINE_RESULTS = os.path.join(ROOT, "evaluation", "results")

# (name, parameter overrides, run clean-up). Each ablation removes one idea from the solution.
SOLUTION_VARIANTS: Tuple[Tuple[str, Dict[str, float], bool], ...] = (
    ("solution", {}, True),
    ("ablate_no_cleanup", {}, False),
    ("ablate_no_smart_tagging", {"tag_team_evidence": 0.0, "set_piece_dead_evidence": 0.0}, True),
    ("ablate_no_team_shape", {"shape_evidence": 0.0}, True),
    ("ablate_no_dead_model", {"out_evidence": 0.0, "motion_evidence": 0.0,
                              "stationary_evidence": 0.0, "set_piece_dead_evidence": 0.0,
                              "moving_ball_evidence": 0.0}, True),
    ("ablate_pre_tuning_priors", {"control_sigma_m": 1.5, "switch_cost": 4.0}, True),
)


def public_gt_codes() -> Optional[np.ndarray]:
    paths = sorted(glob.glob(os.path.join(ROOT, "evaluation", "gt", "*_possession_gt_public.csv")))
    if not paths:
        return None
    return pm.label_codes(pm.read_label_csv(paths[0]))


def baseline_labels(tracking: loaders.Tracking) -> Dict[str, np.ndarray]:
    codes = features.nearest_player_codes(tracking)
    return {name: smooth(codes, window) for name, window in VARIANTS}


def check_against_published(labels: Dict[str, np.ndarray]) -> List[str]:
    """Names of rebuilt baselines that match the CSVs in evaluation/results/ exactly."""
    matched = []
    for name, rebuilt in labels.items():
        path = os.path.join(BASELINE_RESULTS, name + ".csv")
        if not os.path.isfile(path):
            continue
        published = pm.read_label_csv(path)["label"].astype(str).str.strip().values
        if not np.array_equal(published, rebuilt):
            raise AssertionError("rebuilt {} differs from {} -- the vectorised nearest-player "
                                 "step no longer reproduces starter/baseline.py".format(name, path))
        matched.append(name)
    return matched


def row(name: str, labels: np.ndarray, halves, gt: Optional[np.ndarray]) -> Dict[str, object]:
    stats = postprocess.spell_stats(labels, halves, GAMES[GAME]["fps"])
    out: Dict[str, object] = {"name": name, **stats}
    if gt is not None:
        result = pm.evaluate(gt, pm.label_codes(labels))
        out.update(public_macro_f1=result["macro_f1"], public_spell_f1=result["spell_f1_mean"],
                   public_poss=result["n_pred_spells"],
                   public_dur=result["mean_spell_duration_s"])
    return out


def markdown(rows: List[Dict[str, object]], has_gt: bool) -> str:
    head = ["system", "possessions", "mean poss (s)", "stoppages", "H %", "A %", "D %"]
    if has_gt:
        head += ["public macroF1", "public spellF1", "public poss", "public dur (s)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [str(r["name"]), str(r["possessions"]), "{:.1f}".format(r["mean_possession_s"]),
                 str(r["stoppages"]), "{:.1f}".format(100 * r["share_H"]),
                 "{:.1f}".format(100 * r["share_A"]), "{:.1f}".format(100 * r["share_D"])]
        if has_gt:
            cells += ["{:.4f}".format(r["public_macro_f1"]), "{:.4f}".format(r["public_spell_f1"]),
                      str(r["public_poss"]), "{:.1f}".format(r["public_dur"])]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", default=None, help="also write the table to this .md file")
    parser.add_argument("--write-csv", action="store_true",
                        help="write each variant to solution/results/<name>.csv")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args(argv)

    try:
        match = loaders.load_match(GAME, args.data_dir)
    except loaders.DataUnavailableError as error:
        print("ERROR  {}".format(error), file=sys.stderr)
        return 2

    gt = public_gt_codes()
    rows = []
    baselines = baseline_labels(match.tracking)
    matched = check_against_published(baselines)
    for name, labels in baselines.items():
        rows.append(row(name, labels, match.halves, gt))

    for name, overrides, clean in SOLUTION_VARIANTS:
        params = config.DEFAULT_PARAMS.with_overrides(**overrides)
        labels = predict(match, params, clean=clean)
        rows.append(row(name, labels, match.halves, gt))
        if args.write_csv:
            postprocess.write_submission(os.path.join(config.RESULTS_DIR, name + ".csv"), labels)

    table = markdown(rows, gt is not None)
    print(table)
    print("\nBaselines rebuilt from cached tracking; identical to published CSVs: {}".format(
        ", ".join(matched) if matched else "none found to compare (run starter/baseline.py)"))
    if gt is not None:
        print("Public columns: 5 minutes, 17 possessions. A pipeline check, not a ranking -- "
              "see docs/EXPERIMENTS.md.")
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", newline="\n") as handle:
            handle.write(table + "\n")
        print("wrote {}".format(args.report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
