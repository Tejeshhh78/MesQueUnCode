"""Turn an external Metrica sample game into the same MatchData the pipeline runs on.

The two periods are laid out on one timeline with an ungraded gap between them, exactly like the
provided match, so the same decoder, half handling and official metric code apply unchanged.
Labels are derived from the game's own events (labels.py); the clean tracking is degraded towards
the automatic ATD (degrade.py) so ball gaps, noise and dropouts resemble the competition data.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from solution.external import degrade as degradation
from solution.external import labels as labelling
from solution.external import metrica
from solution.loaders import AWAY, HOME, MatchData, Tracking

GAP = 250  # ungraded frames between the periods, so nothing can straddle "halftime"
X_CODE = 3


def _speeds(xy_m: np.ndarray, fps: int) -> np.ndarray:
    """Player speed (m/s) from a centred difference over +/- 2 frames, like the ATD's speed field."""
    speed = np.full(xy_m.shape[:2], np.nan)
    speed[2:-2] = np.linalg.norm(xy_m[4:] - xy_m[:-4], axis=2) * fps / 4.0
    return speed


def _attack_sign(home_xy_m: np.ndarray, first: int, fps: int, length: float) -> int:
    """+1 if the home team attacks towards increasing x in the first period.

    Read from the data, not from labels: a team lines up in its own half at kick-off, so a home
    centroid below halfway in the opening minute means home defends the low-x goal.
    """
    window = home_xy_m[first:first + 60 * fps, :, 0]
    with np.errstate(all="ignore"):
        centroid = np.nanmean(window)
    return 1 if centroid < length / 2.0 else -1


def build(game: metrica.Game, codes: np.ndarray, ball: np.ndarray, home: np.ndarray,
          away: np.ndarray) -> Tuple[MatchData, np.ndarray]:
    """(match, gt_codes): the degraded game as MatchData and its labels on the same timeline.

    Player *positions* are the degraded ones. Player *speeds* are computed from the clean positions
    and then hidden wherever the degraded player is missing: the ATD supplies its speed as its own
    field, and differentiating the degradation's positional jitter would inflate every speed by
    the best part of 1 m/s -- an artefact of the simulation, not of the real tracker.
    """
    length, width = game.pitch
    scale = np.array([length, width])
    pieces_players: List[np.ndarray] = []
    pieces_clean: List[np.ndarray] = []
    pieces_ball: List[np.ndarray] = []
    gt_pieces: List[np.ndarray] = []
    halves = []
    cursor = 0
    n_slots = home.shape[1] + away.shape[1]
    for index, (_, first, last) in enumerate(game.periods):
        if cursor:
            pieces_players.append(np.full((GAP, n_slots, 2), np.nan))
            pieces_clean.append(np.full((GAP, n_slots, 2), np.nan))
            pieces_ball.append(np.full((GAP, 2), np.nan))
            gt_pieces.append(np.full(GAP, X_CODE, dtype=np.int8))
            cursor += GAP
        span = slice(first, last + 1)
        players = np.concatenate([home[span], away[span]], axis=1) * scale
        pieces_players.append(players)
        pieces_clean.append(np.concatenate([game.home_xy[span], game.away_xy[span]], axis=1) * scale)
        pieces_ball.append(ball[span] * scale)
        gt_pieces.append(codes[span].astype(np.int8))
        halves.append((index + 1, cursor, cursor + (last - first)))
        cursor += last - first + 1

    xy = np.concatenate(pieces_players)
    speed = _speeds(np.concatenate(pieces_clean), game.fps)
    players3 = np.concatenate([xy, speed[..., None]], axis=2).astype(np.float32)
    players3[np.isnan(xy).any(axis=2)] = np.nan
    ball3 = np.column_stack([np.concatenate(pieces_ball),
                             np.full(cursor, np.nan)]).astype(np.float32)
    team = np.r_[np.full(home.shape[1], HOME), np.full(away.shape[1], AWAY)].astype(np.int8)
    sign = _attack_sign(game.home_xy * scale, game.periods[0][1], game.fps, length)
    tracking = Tracking(players=players3, ball=ball3, team=team, home_name="Home",
                        away_name="Away", home_attack_sign=sign)
    match = MatchData(tracking=tracking, tags=[], homography_error=None, halves=halves)
    return match, np.concatenate(gt_pieces)


def load_degraded(game_number: int, seed: int,
                  settings: degradation.Degradation = degradation.Degradation()
                  ) -> Tuple[MatchData, np.ndarray, Tuple[float, float]]:
    """One external game, labelled from its events and degraded with a fixed seed."""
    game = metrica.load(game_number)
    codes, _ = labelling.derive(game)
    ball, home, away = degradation.degrade(game, codes, settings, seed=seed)
    match, gt = build(game, codes, ball, home, away)
    return match, gt, tuple(game.pitch)
