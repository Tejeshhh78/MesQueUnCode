"""Per-frame evidence for the decoder: which team could be playing the ball, and is it dead.

Everything here is a log-odds-like score per video frame:

    team[f]  > 0 favours H, < 0 favours A, 0 says nothing
    dead[f]  > 0 favours D, < 0 favours live play, 0 says nothing

Zero is the honest value for "no information" -- the ball is missing on ~41% of graded frames, and
the decoder carries its previous state through a run of zeros, which is exactly the
"possession stays with the last team while the ball is loose" rule of CHALLENGE.md.

No player identities are used anywhere: track slots appear, vanish and swap people, so every
quantity is collective -- the nearest player *of each team*, the mean speed of everyone tracked.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from solution import config
from solution.config import Params
from solution.loaders import AWAY, HOME, MatchData, TagEvent, Tracking


@dataclass
class Evidence:
    team: np.ndarray          # (n,) float64, log-odds H over A
    dead: np.ndarray          # (n,) float64, log-odds D over live
    forced_dead: np.ndarray   # (n,) bool, frames that must be D (before a half's kick-off)
    ball_x: np.ndarray        # (n,) ball after short-gap interpolation, NaN where unknown
    ball_y: np.ndarray
    ball_speed: np.ndarray    # (n,) m/s, NaN where unknown
    dist_home: np.ndarray     # (n,) nearest home player to the ball, m, NaN where unknown
    dist_away: np.ndarray
    kickoffs: List[int]       # detected kick-off frames


# -- small array helpers --------------------------------------------------------------------------
def runs_of(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Half-open [start, stop) spans where mask is True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def centred_mean(values: np.ndarray, halfwidth: int) -> np.ndarray:
    """Mean over a centred +/- halfwidth window, ignoring NaN. NaN where the window is empty."""
    if halfwidth <= 0:
        return values.astype(np.float64).copy()
    valid = ~np.isnan(values)
    total = np.concatenate(([0.0], np.cumsum(np.where(valid, values, 0.0))))
    count = np.concatenate(([0], np.cumsum(valid)))
    index = np.arange(len(values))
    low = np.maximum(0, index - halfwidth)
    high = np.minimum(len(values), index + halfwidth + 1)
    n = count[high] - count[low]
    out = np.full(len(values), np.nan)
    np.divide(total[high] - total[low], n, out=out, where=n > 0)
    return out


def interpolate_short_gaps(values: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly fill NaN runs of length <= max_gap that have an observation on both sides."""
    out = values.astype(np.float64).copy()
    if max_gap <= 0:
        return out
    for start, stop in runs_of(np.isnan(out)):
        if stop - start > max_gap or start == 0 or stop >= len(out):
            continue
        left, right = out[start - 1], out[stop]
        steps = np.arange(1, stop - start + 1) / float(stop - start + 1)
        out[start:stop] = left + (right - left) * steps
    return out


def ball_speed(ball_x: np.ndarray, ball_y: np.ndarray, halfwidth: int, fps: int) -> np.ndarray:
    """Speed from a centred difference over +/- halfwidth frames. NaN if either end is unknown."""
    n = len(ball_x)
    h = max(1, halfwidth)
    speed = np.full(n, np.nan)
    if n <= 2 * h:
        return speed
    dx = ball_x[2 * h:] - ball_x[:-2 * h]
    dy = ball_y[2 * h:] - ball_y[:-2 * h]
    speed[h:n - h] = np.hypot(dx, dy) * fps / (2.0 * h)
    return speed


def nearest_by_team(players: np.ndarray, team: np.ndarray, ball_x: np.ndarray,
                    ball_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(dist_home, dist_away): the nearest tracked player of each team to the ball, NaN if none."""
    dx = players[..., 0] - ball_x[:, None]
    dy = players[..., 1] - ball_y[:, None]
    distance = np.hypot(dx, dy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows are expected: no ball
        home = np.nanmin(np.where((team == HOME)[None, :], distance, np.nan), axis=1)
        away = np.nanmin(np.where((team == AWAY)[None, :], distance, np.nan), axis=1)
    return home.astype(np.float64), away.astype(np.float64)


def nearest_player_codes(tracking: Tracking) -> np.ndarray:
    """The reference baseline's step 1, vectorised: HOME/AWAY of the nearest player, -1 if none.

    Uses the raw ball (no interpolation), exactly like starter/baseline.py, so the experiment
    script can rebuild the published baselines from the cached arrays in seconds.
    """
    home, away = nearest_by_team(tracking.players, tracking.team,
                                 tracking.ball[:, 0].astype(np.float64),
                                 tracking.ball[:, 1].astype(np.float64))
    codes = np.full(len(home), -1, dtype=np.int8)
    known = ~np.isnan(home) | ~np.isnan(away)
    home_filled = np.where(np.isnan(home), np.inf, home)
    away_filled = np.where(np.isnan(away), np.inf, away)
    codes[known] = np.where(home_filled[known] <= away_filled[known], HOME, AWAY)
    return codes


# -- team evidence ------------------------------------------------------------------------------
def control_evidence(dist_home: np.ndarray, dist_away: np.ndarray, speed: np.ndarray,
                     params: Params) -> np.ndarray:
    """Log-odds H over A from who is close enough to be playing the ball, and how fast it moves.

    Each team gets a Gaussian kernel on its nearest player's distance, so "home 0.5 m, away 6 m"
    is strong evidence and "home 4 m, away 5 m" is almost none. A ball in flight is discounted:
    a pass that passes a defender has not been won by him.
    """
    sigma = params.control_sigma_m
    k_home = np.exp(-0.5 * (np.nan_to_num(dist_home, nan=np.inf) / sigma) ** 2)
    k_away = np.exp(-0.5 * (np.nan_to_num(dist_away, nan=np.inf) / sigma) ** 2)
    weight = np.where(np.nan_to_num(speed, nan=0.0) > params.flight_speed,
                      params.flight_weight, 1.0)
    return params.team_evidence * (k_home - k_away) * weight


def blend_across_gaps(evidence: np.ndarray, known: np.ndarray, blend: float,
                      halves: Sequence[Tuple[int, int, int]]) -> np.ndarray:
    """Inside unknown runs, a weak linear blend from the last known evidence to the next.

    Without it a run of zeros is a perfect tie, and where the decoder puts a hand-over inside it
    is an accident of tie-breaking. With it the change lands mid-gap, the least-wrong guess.
    """
    out = evidence.copy()
    if blend <= 0:
        return out
    for _, first, last in halves:
        stop_half = last + 1
        for start, stop in runs_of(~known[first:stop_half]):
            start, stop = start + first, stop + first
            if start == first or stop >= stop_half:
                continue
            before, after = evidence[start - 1], evidence[stop]
            if np.sign(before) == np.sign(after):
                continue  # same side both ends: zeros already carry the state through
            weights = np.arange(1, stop - start + 1) / float(stop - start + 1)
            out[start:stop] = blend * (before * (1 - weights) + after * weights)
    return out


def tag_team_prior(tags: Sequence[TagEvent], n_frames: int, params: Params) -> np.ndarray:
    """Weak log-odds from Smart Tagging phases whose Team is the side in possession."""
    prior = np.zeros(n_frames)
    if params.tag_team_evidence <= 0:
        return prior
    for tag in tags:
        if tag.code not in config.POSSESSION_TAG_CODES or tag.team is None:
            continue
        sign = 1.0 if tag.team == HOME else -1.0
        start, stop = max(0, tag.start_frame), min(n_frames, tag.end_frame)
        prior[start:stop] += sign * params.tag_team_evidence
    return prior


# -- team-shape evidence (model fitted on external games, see calibrate.py) ----------------------
SHAPE_FEATURES = ("centroid_gap", "home_territory", "away_territory", "length_gap", "width_gap",
                  "home_advance", "away_advance")


def team_shape_features(tracking: Tracking, halves: Sequence[Tuple[int, int, int]],
                        params: Params, fps: int,
                        pitch: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """(features, valid): collective team-shape descriptors per frame, signed so that + favours H.

    The tactical facts they encode: the team with the ball stretches the pitch wide and pushes
    forward; the team without it compacts and drops. Every x-quantity is multiplied by the home
    team's attacking direction in that half, so "forward" means the same thing in both halves.
    """
    n = tracking.n_frames
    direction = np.zeros(n)
    for half, first, last in halves:
        direction[first:last + 1] = tracking.home_direction(half)

    x = tracking.players[..., 0].astype(np.float64)
    y = tracking.players[..., 1].astype(np.float64)
    columns = {}
    stats = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for side, code in (("home", HOME), ("away", AWAY)):
            mask = (tracking.team == code)[None, :]
            tx = np.where(mask, x, np.nan)
            ty = np.where(mask, y, np.nan)
            stats[side] = (np.nanmean(tx, axis=1), np.nanstd(tx, axis=1), np.nanstd(ty, axis=1),
                           np.sum(~np.isnan(tx), axis=1))
    lag = fps  # one second
    mid = pitch[0] / 2.0
    for side in ("home", "away"):
        centroid = stats[side][0]
        advance = np.full(n, np.nan)
        advance[lag:] = centroid[lag:] - centroid[:-lag]
        columns[side + "_advance"] = advance * direction
        columns[side + "_territory"] = (centroid - mid) * direction
    columns["centroid_gap"] = (stats["home"][0] - stats["away"][0]) * direction
    columns["length_gap"] = stats["home"][1] - stats["away"][1]
    columns["width_gap"] = stats["home"][2] - stats["away"][2]

    feats = np.column_stack([centred_mean(columns[name], params.shape_halfwidth)
                             for name in SHAPE_FEATURES])
    valid = ((stats["home"][3] >= params.shape_min_players)
             & (stats["away"][3] >= params.shape_min_players)
             & ~np.isnan(feats).any(axis=1) & (direction != 0))
    return feats, valid


def fit_logistic(features_: np.ndarray, labels: np.ndarray, l2: float,
                 iterations: int = 25) -> np.ndarray:
    """Weights (bias first) of an L2-regularised logistic regression, by Newton's method.

    Deterministic and dependency-free; seven features and a few thousand rows converge in a
    handful of steps. Used only by calibrate.py, on external games -- never on the provided match.
    """
    design = np.column_stack([np.ones(len(features_)), features_])
    weights = np.zeros(design.shape[1])
    penalty = l2 * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    for _ in range(iterations):
        prob = 1.0 / (1.0 + np.exp(-design @ weights))
        gradient = design.T @ (prob - labels) + penalty @ weights
        hessian = design.T @ (design * (prob * (1 - prob))[:, None]) + penalty
        step = np.linalg.solve(hessian, gradient)
        weights -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return weights


def shape_log_odds(tracking: Tracking, halves: Sequence[Tuple[int, int, int]], params: Params,
                   fps: int, pitch: Tuple[float, float]) -> np.ndarray:
    """Log-odds H over A from team shape with the frozen external model. 0 where unknown.

    The standardisation and weights come from calibration.json (fitted on other matches); nothing
    is estimated from the match being labelled.
    """
    out = np.zeros(tracking.n_frames)
    if not params.shape_weights:
        return out
    if len(params.shape_weights) != len(SHAPE_FEATURES) + 1:
        raise ValueError("calibration has {} shape weights, expected {}".format(
            len(params.shape_weights), len(SHAPE_FEATURES) + 1))
    feats, valid = team_shape_features(tracking, halves, params, fps, pitch)
    mean = np.asarray(params.shape_mean)
    std = np.asarray(params.shape_std)
    weights = np.asarray(params.shape_weights)
    z = (feats[valid] - mean) / std
    out[valid] = np.clip(weights[0] + z @ weights[1:], -params.shape_clip, params.shape_clip)
    return out


# -- dead-ball evidence -------------------------------------------------------------------------
def out_of_pitch(ball_x: np.ndarray, ball_y: np.ndarray, pitch: Tuple[float, float],
                 margin: float) -> np.ndarray:
    """True where the ball is visibly past a touchline or goal line by more than margin."""
    length, width = pitch
    with np.errstate(invalid="ignore"):
        return ((ball_x < -margin) | (ball_x > length + margin)
                | (ball_y < -margin) | (ball_y > width + margin))


def carry_out_through_gaps(out: np.ndarray, known: np.ndarray) -> np.ndarray:
    """A ball last seen out of play stays out until it is seen again.

    The tracker very often loses the ball as it leaves the pitch (ball boys, crowd, advertising
    boards), so "last seen out" is the best available evidence for the whole gap that follows.
    """
    carried = out.copy()
    for start, stop in runs_of(~known):
        if start > 0 and out[start - 1]:
            carried[start:stop] = True
    return carried


def mean_player_speed(players: np.ndarray, min_players: int) -> np.ndarray:
    """Mean speed of the tracked players, NaN where fewer than min_players are tracked."""
    speeds = players[..., 2].astype(np.float64)
    count = np.sum(~np.isnan(speeds), axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(speeds, axis=1)
    mean[count < min_players] = np.nan
    return mean


def stationary_ball(speed: np.ndarray, params: Params, fps: int) -> np.ndarray:
    """True on runs where the ball stays below stationary_speed for at least stationary_seconds."""
    slow = np.nan_to_num(speed, nan=np.inf) < params.stationary_speed
    out = np.zeros(len(speed), dtype=bool)
    min_len = int(round(params.stationary_seconds * fps))
    for start, stop in runs_of(slow):
        if stop - start >= min_len:
            out[start:stop] = True
    return out


def set_piece_dead_prior(tags: Sequence[TagEvent], n_frames: int, params: Params,
                         fps: int) -> np.ndarray:
    """D evidence over the opening seconds of each Smart Tagging set piece that starts dead."""
    prior = np.zeros(n_frames)
    span = int(round(params.set_piece_dead_seconds * fps))
    for tag in tags:
        if tag.code == "SET PIECES" and tag.set_piece_type in config.DEAD_START_SET_PIECES:
            start = max(0, tag.start_frame)
            prior[start:min(n_frames, start + span)] += params.set_piece_dead_evidence
    return prior


def detect_kickoffs(tags: Sequence[TagEvent], ball_x: np.ndarray, ball_y: np.ndarray,
                    speed: np.ndarray, pitch: Tuple[float, float], params: Params) -> List[int]:
    """Kick-off frames: inside each tagged kick-off window, the first frame the ball leaves the
    centre spot. Falls back to the window's midpoint when the ball is not seen doing it."""
    centre_x, centre_y = pitch[0] / 2.0, pitch[1] / 2.0
    found = []
    for tag in tags:
        if tag.code != "SET PIECES" or tag.set_piece_type not in config.KICKOFF_TYPES:
            continue
        start, stop = max(0, tag.start_frame), min(len(ball_x), tag.end_frame)
        if stop <= start:
            continue
        near = np.hypot(ball_x[start:stop] - centre_x, ball_y[start:stop] - centre_y)
        with np.errstate(invalid="ignore"):
            kicked = (near < params.kickoff_radius_m) & (speed[start:stop] > params.kickoff_speed)
        hits = np.flatnonzero(kicked)
        found.append(start + int(hits[0]) if len(hits) else (start + stop) // 2)
    return sorted(found)


def first_kickoff_per_half(kickoffs: Sequence[int],
                           halves: Sequence[Tuple[int, int, int]]) -> List[Optional[int]]:
    """The earliest detected kick-off inside each half (None if the half has none)."""
    out: List[Optional[int]] = []
    for _, first, last in halves:
        inside = [k for k in kickoffs if first <= k <= last]
        out.append(min(inside) if inside else None)
    return out


# -- assembly -----------------------------------------------------------------------------------
def build_evidence(match: MatchData, params: Params, pitch: Tuple[float, float],
                   fps: int = config.FPS) -> Evidence:
    tracking = match.tracking
    n = tracking.n_frames
    ball_x = interpolate_short_gaps(tracking.ball[:, 0], params.max_interp_gap)
    ball_y = interpolate_short_gaps(tracking.ball[:, 1], params.max_interp_gap)
    known = ~np.isnan(ball_x)
    speed = ball_speed(ball_x, ball_y, params.speed_halfwidth, fps)
    dist_home, dist_away = nearest_by_team(tracking.players, tracking.team, ball_x, ball_y)

    # Team: on-the-ball evidence, smoothed, zero while the ball is out, bridged across gaps.
    out = out_of_pitch(ball_x, ball_y, pitch, params.out_margin_m)
    team = control_evidence(dist_home, dist_away, speed, params)
    team[out] = 0.0
    team = np.nan_to_num(centred_mean(team, params.evidence_window), nan=0.0)
    team = blend_across_gaps(team, known, params.gap_blend, match.halves)
    team += tag_team_prior(match.tags, n, params)

    # Team shape speaks only where the ball cannot: on frames with no ball position.
    if params.shape_evidence > 0:
        shape = shape_log_odds(tracking, match.halves, params, fps, pitch)
        team[~known] += params.shape_evidence * shape[~known]

    # Homography: where the camera-to-pitch fit is poor, positions are less trustworthy.
    if match.homography_error is not None:
        unreliable = np.nan_to_num(match.homography_error, nan=np.inf) > params.homography_max_error
        team[unreliable] *= params.homography_weight

    # Dead ball.
    dead = np.zeros(n)
    dead[carry_out_through_gaps(out, known)] += params.out_evidence
    with np.errstate(invalid="ignore"):
        moving = ~out & (speed > params.moving_ball_speed)
    dead[moving] -= params.moving_ball_evidence
    motion = centred_mean(mean_player_speed(tracking.players, params.min_players_for_motion),
                          params.motion_halfwidth)
    stillness = np.tanh((params.still_speed - motion) / params.still_scale)
    dead += params.motion_evidence * np.nan_to_num(stillness, nan=0.0)
    # Unknown motion is not stillness: NaN compares False, so it never adds dead evidence.
    with np.errstate(invalid="ignore"):
        players_slow = motion < params.still_speed + 0.5
    still_ball = stationary_ball(speed, params, fps) & players_slow
    dead[still_ball] += params.stationary_evidence
    dead += set_piece_dead_prior(match.tags, n, params, fps)

    # Before each half's kick-off the ball is dead by definition (2,934 graded frames in the
    # second half, FAQ.md).
    kickoffs = detect_kickoffs(match.tags, ball_x, ball_y, speed, pitch, params)
    forced = np.zeros(n, dtype=bool)
    for (_, first, _), kickoff in zip(match.halves, first_kickoff_per_half(kickoffs, match.halves)):
        if kickoff is not None:
            forced[first:kickoff] = True

    return Evidence(team=team, dead=dead, forced_dead=forced, ball_x=ball_x, ball_y=ball_y,
                    ball_speed=speed, dist_home=dist_home, dist_away=dist_away,
                    kickoffs=kickoffs)
