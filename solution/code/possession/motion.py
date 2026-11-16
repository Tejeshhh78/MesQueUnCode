"""Collective movement features that mean the same thing on any tracker, and their frozen regimes.

Why: the controlled model's live/dead activity evidence compares mean player speed against two
movement regimes. If those regimes are fitted on one tracker's speeds and applied to another's, a
difference in speed scale (provider speed field vs. speeds differentiated from noisy positions)
breaks it -- measured on external games, spellF1 fell from 0.54 to 0.42. Fitting the regimes on the
provided match itself would avoid that, but the organisers' rule clarification requires anything
fitted to be fitted on external data.

So movement is computed here from player *positions* over a one-second window, by the same code on
the external games and on the provided match. Positional jitter barely moves a one-second
displacement, so the feature keeps its scale across trackers, and regimes fitted once on external
games (``calibrate_activity.py`` -> ``activity_calibration.json``) can be applied unchanged.
"""

import json
from pathlib import Path

import numpy as np

HALF_WINDOW_S = 0.48          # +/- 12 frames at 25 fps: a ~1 s displacement
MAX_PLAYER_SPEED = 15.0       # m/s; faster means a tracking jump, not a player
FAST_PLAYER_SPEED = 2.0       # m/s; "fast" share of players, as in the original feature
CALIBRATION_PATH = Path(__file__).resolve().parent / "activity_calibration.json"


def position_speeds(players_m, fps, half_window_s=0.08):
    """Centred displacement speeds; missing or implausible observations stay unknown."""
    players_m = np.asarray(players_m, dtype=np.float64)
    half = max(1, int(round(half_window_s * fps)))
    speed = np.full(players_m.shape[:2], np.nan)
    if len(players_m) > 2 * half:
        speed[half:-half] = np.linalg.norm(players_m[2 * half:] - players_m[:-2 * half], axis=2) * fps / (2 * half)
    valid = np.isfinite(players_m).all(axis=2) & (speed <= MAX_PLAYER_SPEED)
    return np.where(valid, speed, np.nan)


def apply_position_motion(features, players_m, team, pitch, fps, blocks=None):
    """Use identical movement features for ATD, IDSSE and Metrica observations.

    Keep provider nearest-player speed for the controlled baseline. The optional
    position_player_speed channel is consumed only by the position-motion variant.
    Blocks are inclusive, as in predict(); differences never cross period boundaries.
    """
    n = len(features.ball)
    features.mean_speed = np.full(n, np.nan)
    features.fast_fraction = np.full(n, np.nan)
    features.position_player_speed = np.full((n, 2), np.nan)
    blocks = blocks if blocks is not None else [(1, 0, n - 1)]
    team = np.asarray(team)
    for _, first, last in blocks:
        span = slice(first, last + 1)
        players = np.asarray(players_m[span], dtype=np.float64)
        features.mean_speed[span], features.fast_fraction[span] = robust_motion(players, fps)
        speed = position_speeds(players, fps)
        ball_m = features.ball[span, :2] * np.asarray(pitch)
        distance = np.linalg.norm(players - ball_m[:, None], axis=2)
        for side in (0, 1):
            candidates = np.where((team == side)[None] & np.isfinite(distance), distance, np.inf)
            nearest = np.argmin(candidates, axis=1)
            rows = np.arange(len(players))
            found = np.isfinite(candidates[rows, nearest])
            features.position_player_speed[first + rows[found], side] = speed[rows[found], nearest[found]]
    return features


def robust_motion(players_m, fps):
    """(mean_speed, fast_fraction) per frame from player positions in metres, shape (n, slots, 2).

    Speed of each player = displacement over +/- HALF_WINDOW_S divided by the elapsed time; NaN
    where the player is not seen at both ends or is not seen now.
    """
    players_m = np.asarray(players_m, dtype=np.float64)
    n = players_m.shape[0]
    half = max(1, int(round(HALF_WINDOW_S * fps)))
    speed = np.full(players_m.shape[:2], np.nan)
    if n > 2 * half:
        step = np.linalg.norm(players_m[2 * half:] - players_m[:-2 * half], axis=2)
        speed[half:n - half] = step * fps / (2.0 * half)
    speed[~np.isfinite(players_m).all(axis=2)] = np.nan
    valid = np.isfinite(speed) & (speed <= MAX_PLAYER_SPEED)
    count = valid.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, np.where(valid, speed, 0.0).sum(axis=1) / np.maximum(count, 1), np.nan)
        fast = np.where(count > 0, (valid & (speed > FAST_PLAYER_SPEED)).sum(axis=1) / np.maximum(count, 1),
                        np.nan)
    return mean, fast


def load_regimes(path=CALIBRATION_PATH):
    """(means, variances, weights) of the two movement regimes, fitted on external games only."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. It is committed with the code; regenerate it from Metrica's public "
            "sample games with:  python -m solution.code.calibrate_activity --write")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("provenance", {}).get("provided_match_used") is not False:
        raise ValueError(f"{path} does not declare external-only provenance")
    regimes = data["regimes"]
    return (np.asarray(regimes["means"], dtype=np.float64),
            np.asarray(regimes["variances"], dtype=np.float64),
            np.asarray(regimes["weights"], dtype=np.float64))
