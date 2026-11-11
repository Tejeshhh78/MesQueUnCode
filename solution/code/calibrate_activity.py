"""Fit the controlled model's movement regimes on EXTERNAL games, and score it there.

    python -m solution.code.calibrate_activity --write     # fit on all games, write the JSON
    python -m solution.code.calibrate_activity --check     # refit and confirm the saved JSON

The organisers' rule clarification requires anything fitted to be fitted on external data. The
controlled model's only fitted part is its two movement regimes (model.fit_regimes). Here they are
fitted on Metrica Sports' three public sample games -- labels derived from each game's own events,
tracking degraded towards the competition's automatic ATD (solution/external/, Leo Hofmann's
harness) -- using exactly the movement feature used at inference (motion.robust_motion). The
result, possession/activity_calibration.json, is only applied on the provided match.

It also prints a leave-one-game-out evaluation with the official metric code: each game is scored
with regimes fitted on the other two. The provided BAR-ATH match is never read here.
"""

import argparse
import json
import os
import sys
from functools import partial

os.environ.setdefault("MQUC_CALIBRATING", "1")

import numpy as np  # noqa: E402

from evaluation import possession_metrics as pm  # noqa: E402
from solution.code.possession import model  # noqa: E402
from solution.code.possession.data import Features  # noqa: E402
from solution.code.possession.motion import CALIBRATION_PATH, robust_motion  # noqa: E402
from solution.external import build, degrade  # noqa: E402
from solution import features as solution_features  # noqa: E402
from starter.baseline import smooth  # noqa: E402

GAMES = (1, 2, 3)
SEEDS = (0, 1)
FPS = 25


def external_features(match, pitch):
    """controlled-model Features for an external game, built as at inference.

    Distances and nearest-player speeds follow Leo's harness (validation/harness.py); movement
    (mean_speed, fast_fraction) comes from motion.robust_motion on the degraded positions, exactly
    as __main__ computes it from the provided match's positions.
    """
    tracking = match.tracking
    scale = np.asarray(pitch, dtype=np.float64)
    players_m = tracking.players[..., :2].astype(np.float64)
    ball = tracking.ball[:, :2].astype(np.float64) / scale
    n = len(ball)
    visible = np.isfinite(players_m).all(axis=2)
    short = np.full(players_m.shape[:2], np.nan)
    short[2:-2] = np.linalg.norm(players_m[4:] - players_m[:-4], axis=2) * FPS / 4
    delta = players_m - (ball * scale)[:, None, :]
    distance_all = np.where(visible, np.hypot(delta[..., 0], delta[..., 1]), np.inf)
    ball_visible = np.isfinite(ball).all(axis=1)
    distance = np.full((n, 2), np.nan)
    player_speed = np.full((n, 2), np.nan)
    for side in (0, 1):
        masked = np.where((tracking.team == side)[None, :], distance_all, np.inf)
        nearest = np.argmin(masked, axis=1)
        minimum = masked[np.arange(n), nearest]
        ok = ball_visible & np.isfinite(minimum)
        distance[ok, side] = minimum[ok]
        player_speed[ok, side] = short[np.flatnonzero(ok), nearest[ok]]
    mean_speed, fast = robust_motion(players_m, FPS)
    return Features(np.column_stack([ball, np.zeros(n)]), distance, player_speed, mean_speed, fast,
                    visible.sum(axis=1).astype(np.int16), ("Home", "Away"))


def load_games():
    out = {}
    for game in GAMES:
        for seed in SEEDS:
            match, gt, pitch = build.load_degraded(game, seed)
            blocks = [(h, a, b) for h, a, b in match.halves]
            out[(game, seed)] = (external_features(match, pitch), gt, blocks, pitch, match)
    return out


def samples(features, blocks):
    mask = np.zeros(len(features.ball), dtype=bool)
    for _, first, last in blocks:
        mask[first:last + 1] = True
    x = np.log1p(np.clip(features.mean_speed, 0, 15))[mask][::FPS]
    return x[np.isfinite(x)]


def fit(games, keys):
    return model.fit_regimes(np.concatenate([samples(games[k][0], games[k][2]) for k in keys]))


def score(gt, labels):
    labels = np.asarray(labels).copy()
    labels[gt == 3] = "X"
    result = pm.evaluate(gt, pm.label_codes(labels))
    return result["spell_f1_mean"], result["macro_f1"], result["n_pred_spells"] / result["n_gt_spells"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--no-eval", action="store_true", help="skip the leave-one-game-out scores")
    args = parser.parse_args(argv)
    games = load_games()

    if not args.no_eval:
        print("| game | seed | baseline_smooth50 spellF1 | controlled spellF1 | controlled macroF1 | poss vs truth |")
        print("|---:|---:|---:|---:|---:|---:|")
        rows = []
        original = model.activity_evidence
        try:
            for (game, seed), (features, gt, blocks, pitch, match) in games.items():
                regimes = fit(games, [k for k in games if k[0] != game])
                model.activity_evidence = partial(original, regimes=regimes)
                labels, _ = model.predict(features, [], blocks, pitch, FPS, profile="controlled")
                ours = score(gt, labels)
                base = smooth(solution_features.nearest_player_codes(match.tracking), 50)
                baseline = score(gt, base)
                rows.append((baseline[0], *ours))
                print("| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.2f}x |".format(
                    game, seed, baseline[0], *ours))
        finally:
            model.activity_evidence = original
        means = np.mean(rows, axis=0)
        print("mean: baseline spellF1 {:.4f}; controlled spellF1 {:.4f}, macroF1 {:.4f}, "
              "{:.2f}x the true possessions".format(*means))

    means, variances, weights = fit(games, list(games))
    calibration = {
        "regimes": {"means": [round(float(v), 6) for v in means],
                    "variances": [round(float(v), 6) for v in variances],
                    "weights": [round(float(v), 6) for v in weights]},
        "feature": "log1p(mean player speed over a +/-0.48 s displacement), motion.robust_motion",
        "provenance": {
            "data": "Metrica Sports public sample games 1, 2, 3 "
                    "(https://github.com/metrica-sports/sample-data)",
            "labels": "not used for fitting (unsupervised EM); event-derived labels only for scoring",
            "tracking": "degraded towards the automatic ATD by solution/external/degrade.py",
            "seeds": list(SEEDS),
            "provided_match_used": False,
        },
    }
    print("\nregime speeds (m/s):", [round(float(v), 3) for v in np.expm1(means)],
          "weights:", calibration["regimes"]["weights"])
    if args.check:
        saved = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
        same = all(np.allclose(saved["regimes"][k], calibration["regimes"][k], atol=1e-5)
                   for k in calibration["regimes"])
        print("check:", "matches saved calibration" if same else "DIFFERS from saved calibration")
        if not same:
            return 1
    if args.write:
        with CALIBRATION_PATH.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(calibration, handle, indent=2)
            handle.write("\n")
        print("wrote", CALIBRATION_PATH.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
