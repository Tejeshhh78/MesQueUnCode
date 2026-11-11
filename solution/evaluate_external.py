"""Score the whole pipeline on EXTERNAL games with the official metric code.

    python solution/evaluate_external.py

This is the model's only legitimate quality signal: the provided match's full ground truth is
private, and the organisers forbid tuning on the provided match or its public window. Each of
Metrica's three public sample games is scored with a calibration fitted on the *other two*
(leave-one-game-out), so no game is scored by a model that saw it. The starter's
baseline_smooth50 is scored on exactly the same degraded tracking for reference.

The motion/shape calibration is leave-one-game-out here; the decoder costs are the ones in
config.py, which tune_external.py chose on all three games, so these scores are slightly
optimistic. tune_external.py's nested leave-one-game-out figure is the honest one.

The provided BAR-ATH match is never read by this script.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

os.environ.setdefault("MQUC_CALIBRATING", "1")

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from solution import calibrate, config, features  # noqa: E402
from solution.external import build  # noqa: E402
from solution.run_solution import predict  # noqa: E402
from starter.baseline import smooth  # noqa: E402

LETTERS = np.array(["H", "A", "D", "X"])


def params_from(calibration: Dict[str, Dict[str, object]]) -> config.Params:
    return config.Params().with_overrides(
        still_speed=calibration["motion"]["still_speed"],
        still_scale=calibration["motion"]["still_scale"],
        shape_mean=tuple(calibration["shape"]["mean"]),
        shape_std=tuple(calibration["shape"]["std"]),
        shape_weights=tuple(calibration["shape"]["weights"]))


def baseline_labels(match, gt: np.ndarray) -> np.ndarray:
    labels = smooth(features.nearest_player_codes(match.tracking), 50).astype("<U1")
    labels[gt == calibrate.X] = "X"
    return labels


def score(gt: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    result = pm.evaluate(gt, pm.label_codes(labels))
    return {"spellF1": result["spell_f1_mean"], "macroF1": result["macro_f1"],
            "poss": result["n_pred_spells"], "true_poss": result["n_gt_spells"]}


def main() -> int:
    base = config.Params()
    rows = {(g, s): calibrate.game_rows(g, s, base) for g in calibrate.GAMES for s in calibrate.SEEDS}
    table: List[Dict[str, object]] = []
    for held in calibrate.GAMES:
        calibration = calibrate.fit([rows[k] for k in rows if k[0] != held])
        params = params_from(calibration)
        for seed in calibrate.SEEDS:
            match, gt, pitch = build.load_degraded(held, seed)
            ours = score(gt, predict(match, params, pitch=pitch))
            baseline = score(gt, baseline_labels(match, gt))
            table.append({"game": held, "seed": seed, "ours": ours, "baseline": baseline})

    print("| game | seed | true poss | baseline_smooth50 spellF1 | ours spellF1 | baseline macroF1 | ours macroF1 | ours poss |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in table:
        o, b = row["ours"], row["baseline"]
        print("| {} | {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {} |".format(
            row["game"], row["seed"], o["true_poss"], b["spellF1"], o["spellF1"],
            b["macroF1"], o["macroF1"], o["poss"]))
    for key in ("spellF1", "macroF1"):
        ours = np.mean([r["ours"][key] for r in table])
        baseline = np.mean([r["baseline"][key] for r in table])
        print("mean {}: baseline {:.4f}, ours {:.4f}".format(key, baseline, ours))
    ratio = np.mean([r["ours"]["poss"] / r["ours"]["true_poss"] for r in table])
    print("ours predicts {:.2f}x the true number of possessions on average".format(ratio))
    return 0


if __name__ == "__main__":
    sys.exit(main())
