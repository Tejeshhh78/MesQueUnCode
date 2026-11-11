"""Deterministic degradation of clean tracking towards the challenge's automatic ATD.

The Metrica sample games are clean: the ball is tracked on ~96 % of live frames and almost never
while play is stopped. The challenge ATD is automatic: the ball is missing on 41 % of graded frames,
disproportionately in crowded or aerial moments (DATA.md), yet it is often visible while play is
stopped (placed for a restart, held for a throw-in, rolling outside the pitch). Without modelling
that gap, a validation run would reward "ball missing means dead", which fails on the real ATD.

Everything is a seeded function of the clean game and the dead-ball intervals of its derived labels.
The settings are assumptions about the automatic tracker, not measurements; sweep them.
"""

from dataclasses import dataclass, replace

import numpy as np

from .labels import D


@dataclass(frozen=True)
class Degradation:
    alive_missing: float = 0.35    # target share of live frames without a ball observation
    mean_gap_s: float = 1.2        # mean length of one ball dropout
    crowd_factor: float = 3.0      # relative dropout hazard with >= 3 players within 3 m
    fast_factor: float = 2.0       # relative dropout hazard for balls faster than 14 m/s
    dead_visible: float = 0.4      # mean share of dead frames with a synthesised ball
    ball_noise_m: float = 0.25     # AR(1) positional noise, stationary standard deviation
    player_noise_m: float = 0.2
    player_missing: float = 0.07   # challenge: 20.5 of 22 players visible on average
    outlier_rate: float = 0.002    # spurious ball jumps per live frame

    def with_(self, **changes):
        return replace(self, **changes)


def _ar1_noise(rng, shape, sigma, rho=0.9):
    """Smooth noise: tracking jitter is temporally correlated, not white."""
    innovations = rng.normal(0.0, sigma * np.sqrt(1 - rho ** 2), size=shape)
    out = np.empty(shape)
    out[0] = rng.normal(0.0, sigma, size=shape[1:])
    for t in range(1, shape[0]):
        out[t] = rho * out[t - 1] + innovations[t]
    return out


def _markov_hidden(hazard, recover, u_lose, u_back):
    hidden = np.zeros(len(hazard), dtype=bool)
    state = False
    for t in range(len(hazard)):
        state = (u_back[t] >= recover) if state else (u_lose[t] < hazard[t])
        hidden[t] = state
    return hidden


def _segments(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def degrade(game, codes, settings=Degradation(), seed=0):
    """Return (ball_xy, home_xy, away_xy) degraded copies, normalised coordinates."""
    rng = np.random.default_rng(seed)
    pitch = np.asarray(game.pitch, dtype=np.float64)
    fps = game.fps
    n = len(game.ball_xy)
    ball_true = game.ball_xy.copy()
    alive = (codes == 0) | (codes == 1)
    dead = codes == D

    # Context of the ball: crowding and speed (on the clean data, i.e. what the camera faced).
    players = np.concatenate([game.home_xy, game.away_xy], axis=1)
    delta = (players - ball_true[:, None, :]) * pitch
    near = np.sum(np.nan_to_num(np.hypot(delta[..., 0], delta[..., 1]), nan=np.inf) < 3.0, axis=1)
    metres = ball_true * pitch
    speed = np.full(n, np.nan)
    speed[2:-2] = np.hypot(*(metres[4:] - metres[:-4]).T) * fps / 4
    weight = 1 + settings.crowd_factor * (near >= 3) + settings.fast_factor * (np.nan_to_num(speed) > 14)

    # Ball dropouts on live frames, calibrated to the target missing share by bisection.
    recover = 1.0 / max(1.0, settings.mean_gap_s * fps)
    u_lose, u_back = rng.random(n), rng.random(n)
    clean_missing = ~np.isfinite(ball_true).all(axis=1)
    low, high = 0.0, 1.0
    hidden = np.zeros(n, dtype=bool)
    if settings.alive_missing > 0 and alive.any():
        for _ in range(14):
            middle = (low + high) / 2
            hidden = _markov_hidden(np.where(alive, np.minimum(1, middle * weight), 0.0),
                                    recover, u_lose, u_back)
            share = np.mean((hidden | clean_missing)[alive])
            low, high = (middle, high) if share < settings.alive_missing else (low, middle)
    ball = ball_true + _ar1_noise(rng, (n, 2), settings.ball_noise_m) / pitch
    ball[hidden & alive] = np.nan

    # Spurious detections: short jumps of 3-15 m on live frames.
    for start in np.flatnonzero(alive & (rng.random(n) < settings.outlier_rate)):
        length = int(rng.integers(1, 4))
        angle = rng.uniform(0, 2 * np.pi)
        jump = rng.uniform(3, 15) * np.array([np.cos(angle), np.sin(angle)]) / pitch
        ball[start:start + length] = ball[start:start + length] + jump

    # Dead ball: the clean data rarely tracks it, the automatic tracker often does. Synthesise
    # the ball continuing past the line, then lying at the restart spot before the restart.
    ball[dead] = np.nan
    for start, stop in _segments(dead):
        length = stop - start
        before = ball_true[max(0, start - 4):start]
        before = before[np.isfinite(before).all(axis=1)]
        exit_len = min(length, int(0.5 * fps))
        if len(before) >= 2 and exit_len:
            velocity = (before[-1] - before[0]) / max(1, len(before) - 1)
            decay = np.cumprod(np.full(exit_len, 0.9))
            ball[start:start + exit_len] = before[-1] + np.cumsum(decay)[:, None] * velocity
        after = ball_true[stop:min(n, stop + 10)]
        after = after[np.isfinite(after).all(axis=1)]
        share = np.clip(rng.normal(settings.dead_visible, 0.15), 0, 1)
        place = int(min(length - exit_len, max(0, round(share * length) - exit_len)))
        if place and len(after):
            jitter = rng.normal(0, 0.1, size=(place, 2)) / pitch
            ball[stop - place:stop] = after[0] + jitter

    def players_degraded(xy):
        out = xy + _ar1_noise(rng, xy.shape, settings.player_noise_m) / pitch
        gap = 2.0 * fps
        visible_mean = gap * (1 - settings.player_missing) / max(1e-9, settings.player_missing)
        for slot in range(xy.shape[1]):
            # Alternating renewal process: geometric visible and hidden run lengths.
            t = int(rng.integers(0, int(visible_mean) + 1))
            while t < n:
                length = int(rng.geometric(1 / gap))
                out[t:t + length, slot] = np.nan
                t += length + int(rng.geometric(1 / visible_mean))
        out[~np.isfinite(xy)] = np.nan
        return out

    return ball, players_degraded(game.home_xy), players_degraded(game.away_xy)
