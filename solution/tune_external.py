"""Tune the decoder's few free parameters on EXTERNAL games only.

    python solution/tune_external.py

The organisers' rule clarification requires every tuned parameter to be tuned on external data.
This script grid-searches the hand-over cost, the minimum possession length and the control radius
on Metrica's three public sample games (labels from their own events, tracking degraded towards
the automatic ATD), scoring with the official possession_metrics code.

It reports two things:
  * nested leave-one-game-out: for each game, the grid point chosen on the OTHER two games and its
    score on the held-out game -- an honest estimate of what tuning buys on an unseen match;
  * the grid point best on all three games, which is what solution/config.py uses.

The provided BAR-ATH match is never read by this script.
"""

from __future__ import annotations

import itertools
import os
import sys
from typing import Dict, Tuple

os.environ.setdefault("MQUC_CALIBRATING", "1")

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from solution import calibrate, config, features, possession_model, postprocess  # noqa: E402
from solution.evaluate_external import params_from  # noqa: E402
from solution.external import build  # noqa: E402

FPS = 25
GRID = {
    "control_sigma_m": (0.6, 0.8, 1.0, 1.5),
    "switch_cost": (0.25, 0.5, 1.0, 1.5, 2.0),
    "min_spell_seconds": (0.7, 1.0),
}
# The hand-picked priors the model used before any external tuning, kept as the comparison point.
REFERENCE = {"control_sigma_m": 1.5, "switch_cost": 4.0, "min_spell_seconds": 1.0}


def main() -> int:
    base = config.Params()
    rows = {(g, s): calibrate.game_rows(g, s, base) for g in calibrate.GAMES for s in calibrate.SEEDS}
    # Calibration (motion + shape) always leave-one-game-out, so the scored game never shaped it.
    held_params = {g: params_from(calibrate.fit([rows[k] for k in rows if k[0] != g]))
                   for g in calibrate.GAMES}
    data = {(g, s): build.load_degraded(g, s) for g in calibrate.GAMES for s in calibrate.SEEDS}

    points = [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]
    points.append(dict(REFERENCE))  # scored for comparison; it competes like any other point
    scores: Dict[Tuple[int, Tuple], float] = {}
    for (game, seed), (match, gt, pitch) in data.items():
        for sigma in sorted({p["control_sigma_m"] for p in points}):
            params = held_params[game].with_overrides(control_sigma_m=sigma)
            evidence = features.build_evidence(match, params, pitch, FPS)
            for point in (p for p in points if p["control_sigma_m"] == sigma):
                tuned = params.with_overrides(**point)
                states = possession_model.decode(evidence, tuned, match.halves)
                states = postprocess.clean(states, match.halves,
                                           int(round(tuned.min_spell_seconds * FPS)),
                                           int(round(tuned.min_dead_seconds * FPS)))
                labels = postprocess.to_labels(states)
                labels[gt == calibrate.X] = "X"
                result = pm.evaluate(gt, pm.label_codes(labels))
                key = tuple(sorted(point.items()))
                scores[(game, seed, key)] = result["spell_f1_mean"]

    def game_mean(key, games):
        return float(np.mean([scores[(g, s, key)] for g in games for s in calibrate.SEEDS]))

    keys = [tuple(sorted(p.items())) for p in points]
    default_key = tuple(sorted(REFERENCE.items()))

    print("nested leave-one-game-out (choose on two games, score the third):")
    held_scores = []
    for held in calibrate.GAMES:
        others = [g for g in calibrate.GAMES if g != held]
        best = max(keys, key=lambda k: (game_mean(k, others), k))
        held_scores.append(game_mean(best, [held]))
        print("  game {}: chosen {} -> held-out spellF1 {:.4f} (pre-tuning priors {:.4f})".format(
            held, dict(best), held_scores[-1], game_mean(default_key, [held])))
    print("  mean held-out spellF1 with tuning {:.4f}, with pre-tuning priors {:.4f}".format(
        np.mean(held_scores), np.mean([game_mean(default_key, [g]) for g in calibrate.GAMES])))

    ranked = sorted(keys, key=lambda k: -game_mean(k, calibrate.GAMES))
    print("\nbest grid points on all three games:")
    for key in ranked[:5]:
        print("  {:.4f}  {}".format(game_mean(key, calibrate.GAMES), dict(key)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
