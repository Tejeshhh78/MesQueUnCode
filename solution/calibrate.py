"""Fit the model's learned parts on EXTERNAL games and write solution/calibration.json.

    python solution/calibrate.py --download --write     # first time: fetch the games, fit, save
    python solution/calibrate.py --check                 # refit and confirm the saved file matches

Why this exists: the organisers' rule clarification says anything trained, fitted, calibrated or
tuned must use external data only. Two parts of the model are fitted, and both are fitted here, on
Metrica Sports' three public sample games -- other matches, with labels derived mechanically from
their own event data (solution/external/labels.py) and tracking degraded towards the automatic ATD
(solution/external/degrade.py):

  * motion   a logistic regression of "dead ball" on smoothed mean player speed, giving the
             centre (still_speed) and steepness (still_scale) of the dead-ball motion evidence;
  * shape    a logistic regression of "home has the ball" on seven collective team-shape
             features, with its feature standardisation.

The provided BAR-ATH match is never read by this script. On the provided match the fitted values
are only applied.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import Dict, Sequence

os.environ.setdefault("MQUC_CALIBRATING", "1")  # config may load before calibration.json exists

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from solution import config, features  # noqa: E402
from solution.external import build, degrade, metrica  # noqa: E402

H, A, D, X = 0, 1, 2, 3
GAMES = (1, 2, 3)
SEEDS = (0, 1)
SUBSAMPLE = 5        # use every 5th frame: neighbouring frames carry almost the same information
MOTION_L2 = 1e-3
SHAPE_L2 = 1.0
FPS = 25


def game_rows(game: int, seed: int, params: config.Params) -> Dict[str, np.ndarray]:
    """Motion and shape training rows from one degraded external game."""
    match, gt, pitch = build.load_degraded(game, seed)
    tracking = match.tracking
    motion = features.centred_mean(
        features.mean_player_speed(tracking.players, params.min_players_for_motion),
        params.motion_halfwidth)
    graded = gt != X
    keep = np.zeros(len(gt), dtype=bool)
    keep[::SUBSAMPLE] = True
    motion_rows = keep & graded & np.isfinite(motion)

    feats, valid = features.team_shape_features(tracking, match.halves, params, FPS, pitch)
    alive = (gt == H) | (gt == A)
    shape_rows = keep & alive & valid
    ball_missing = np.isnan(tracking.ball[:, 0])
    return {
        "motion_x": motion[motion_rows], "motion_y": (gt[motion_rows] == D).astype(float),
        "shape_x": feats[shape_rows], "shape_y": (gt[shape_rows] == H).astype(float),
        "shape_gap": ball_missing[shape_rows],
    }


def fit(rows: Sequence[Dict[str, np.ndarray]]) -> Dict[str, Dict[str, object]]:
    """Calibration dict from training rows of one or more games."""
    mx = np.concatenate([r["motion_x"] for r in rows])
    my = np.concatenate([r["motion_y"] for r in rows])
    intercept, slope = features.fit_logistic(mx[:, None], my, MOTION_L2)
    if slope >= 0:
        raise ValueError("dead-ball probability does not fall with player speed in the external "
                         "data (slope {:.3f}); refusing to write a nonsensical calibration".format(slope))
    # p(dead) = sigmoid(intercept + slope * v) and tanh(z / 2) = 2 * sigmoid(z) - 1, so the model's
    # tanh((still_speed - v) / still_scale) has still_speed = -intercept / slope and
    # still_scale = -2 / slope.
    still_speed = -intercept / slope
    still_scale = -2.0 / slope

    sx = np.concatenate([r["shape_x"] for r in rows])
    sy = np.concatenate([r["shape_y"] for r in rows])
    mean = sx.mean(axis=0)
    std = sx.std(axis=0)
    std[std == 0] = 1.0
    weights = features.fit_logistic((sx - mean) / std, sy, SHAPE_L2)
    return {
        "motion": {"still_speed": float(still_speed), "still_scale": float(still_scale)},
        "shape": {"features": list(features.SHAPE_FEATURES), "mean": mean.tolist(),
                  "std": std.tolist(), "weights": weights.tolist()},
    }


def held_out_scores(calibration: Dict[str, Dict[str, object]],
                    rows: Sequence[Dict[str, np.ndarray]]) -> Dict[str, float]:
    """Accuracy of both fitted models on rows they were not fitted on, against the majority class."""
    mx = np.concatenate([r["motion_x"] for r in rows])
    my = np.concatenate([r["motion_y"] for r in rows])
    m = calibration["motion"]
    dead_pred = np.tanh((m["still_speed"] - mx) / m["still_scale"]) > 0
    sx = np.concatenate([r["shape_x"] for r in rows])
    sy = np.concatenate([r["shape_y"] for r in rows])
    gap = np.concatenate([r["shape_gap"] for r in rows])
    s = calibration["shape"]
    logit = s["weights"][0] + ((sx - np.asarray(s["mean"])) / np.asarray(s["std"])) @ np.asarray(s["weights"][1:])
    home_pred = logit > 0
    return {
        "motion_accuracy": float(np.mean(dead_pred == (my == 1))),
        "motion_majority": float(max(my.mean(), 1 - my.mean())),
        "shape_accuracy": float(np.mean(home_pred == (sy == 1))),
        "shape_accuracy_ball_missing": float(np.mean(home_pred[gap] == (sy[gap] == 1))) if gap.any() else float("nan"),
        "shape_majority": float(max(sy.mean(), 1 - sy.mean())),
    }


def rounded(calibration: Dict[str, Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    def r(value):
        if isinstance(value, float):
            return round(value, 6)
        if isinstance(value, list):
            return [r(v) for v in value]
        if isinstance(value, dict):
            return {k: r(v) for k, v in value.items()}
        return value
    return r(calibration)


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--download", action="store_true", help="fetch the public sample games first")
    parser.add_argument("--write", action="store_true", help="write solution/calibration.json")
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if a refit differs from the saved calibration")
    args = parser.parse_args(argv)
    if args.download:
        metrica.download(GAMES)

    params = config.Params()  # only design constants are read here (windows, minimum players)
    rows = {(g, s): game_rows(g, s, params) for g in GAMES for s in SEEDS}

    print("leave-one-game-out (fit on the other two games, score the held-out one):")
    print("  game  motion acc (majority)   shape acc  [ball missing]  (majority)")
    for held in GAMES:
        train = [rows[k] for k in rows if k[0] != held]
        test = [rows[k] for k in rows if k[0] == held]
        scores = held_out_scores(fit(train), test)
        print("  {:>4}  {motion_accuracy:.3f} ({motion_majority:.3f})        {shape_accuracy:.3f}  "
              "[{shape_accuracy_ball_missing:.3f}]         ({shape_majority:.3f})".format(held, **scores))

    calibration = rounded(fit(list(rows.values())))
    calibration["provenance"] = {
        "data": "Metrica Sports public sample games 1, 2, 3 "
                "(https://github.com/metrica-sports/sample-data)",
        "labels": "derived from each game's own event data by solution/external/labels.py",
        "tracking": "degraded towards the automatic ATD by solution/external/degrade.py",
        "seeds": list(SEEDS),
        "degradation": asdict(degrade.Degradation()),
        "subsample": SUBSAMPLE,
        "provided_match_used": False,
    }
    print("\nfitted on all external games:")
    print("  still_speed {still_speed:.3f} m/s, still_scale {still_scale:.3f}".format(**calibration["motion"]))
    print("  shape weights", calibration["shape"]["weights"])

    if args.check:
        if not os.path.isfile(config.CALIBRATION_PATH):
            print("no saved calibration to check against", file=sys.stderr)
            return 1
        with open(config.CALIBRATION_PATH, encoding="utf-8") as handle:
            saved = json.load(handle)
        same = all(np.allclose(np.asarray(saved[k][f], dtype=float),
                               np.asarray(calibration[k][f], dtype=float), atol=1e-5)
                   for k in ("motion", "shape") for f in calibration[k] if f != "features")
        print("check: {}".format("matches saved calibration" if same else "DIFFERS from saved calibration"))
        if not same:
            return 1
    if args.write:
        with open(config.CALIBRATION_PATH, "w", newline="\n", encoding="utf-8") as handle:
            json.dump(calibration, handle, indent=2)
            handle.write("\n")
        print("wrote {}".format(os.path.relpath(config.CALIBRATION_PATH, ROOT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
