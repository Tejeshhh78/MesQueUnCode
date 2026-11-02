"""Synthetic matches for the tests: small, hand-built, and needing none of the gated data."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from solution.loaders import AWAY, HOME, MatchData, TagEvent, Tracking

PITCH = (105.3, 68.0)
# Two short "halves" with a halftime gap and ungraded frames either side, mirroring the real layout.
HALVES: List[Tuple[int, int, int]] = [(1, 50, 449), (2, 600, 999)]
N_FRAMES = 1100


def make_tracking(n_frames: int = N_FRAMES) -> Tracking:
    """Four slots (two per team) standing still in midfield, ball missing everywhere."""
    players = np.full((n_frames, 4, 3), np.nan, dtype=np.float32)
    positions = [(40.0, 30.0), (45.0, 40.0), (60.0, 30.0), (65.0, 40.0)]
    for slot, (x, y) in enumerate(positions):
        players[:, slot, 0] = x
        players[:, slot, 1] = y
        players[:, slot, 2] = 2.0  # jogging: neither still nor sprinting
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    team = np.array([HOME, HOME, AWAY, AWAY], dtype=np.int8)
    return Tracking(players=players, ball=ball, team=team, home_name="Home FC",
                    away_name="Away FC", home_attack_sign=1)


def put_ball_at_player(tracking: Tracking, slot: int, start: int, stop: int,
                       offset: float = 0.3) -> None:
    """Player dribbles (weaving at up to ~5 m/s) with the ball at his feet on [start, stop)."""
    steps = np.arange(stop - start)
    tracking.players[start:stop, slot, 1] += 3.0 * np.sin(2 * np.pi * steps / 100.0)
    tracking.ball[start:stop, 0] = tracking.players[start:stop, slot, 0] + offset
    tracking.ball[start:stop, 1] = tracking.players[start:stop, slot, 1]


def make_match(tracking: Optional[Tracking] = None, tags: Sequence[TagEvent] = (),
               halves: Sequence[Tuple[int, int, int]] = HALVES) -> MatchData:
    return MatchData(tracking=tracking or make_tracking(), tags=list(tags),
                     homography_error=None, halves=list(halves))


def possession_script() -> MatchData:
    """First half: home on the ball, then away; second half: home, ball out, away restarts."""
    tracking = make_tracking()
    put_ball_at_player(tracking, 0, 50, 200)     # home controls
    put_ball_at_player(tracking, 2, 230, 449)    # away wins it
    put_ball_at_player(tracking, 1, 600, 800)    # home again after halftime
    tracking.ball[800:900, 0] = 50.0             # ball out over the touchline
    tracking.ball[800:900, 1] = 75.0
    tracking.players[800:900, :, 2] = 0.5        # and everyone walks
    put_ball_at_player(tracking, 3, 900, 1000)   # away restarts
    return make_match(tracking)
