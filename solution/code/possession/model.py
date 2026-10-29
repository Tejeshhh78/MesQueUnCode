"""Label-independent evidence and exact temporal state inference.

Incumbent scales and regularisation were fixed before the public pipeline check.
Explicit experiments use independent other-match validation, never this match's labels.
"""

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher

import numpy as np

from .data import normalise_name


@dataclass(frozen=True)
class Config:
    control_radius_m: float = 6.0
    distance_margin_m: float = 2.5
    short_gap_s: float = 0.32
    motion_window_s: float = 1.0
    min_state_s: float = 0.32
    team_switch_cost: float = 20.0
    stoppage_switch_cost: float = 24.0
    tag_strength: float = 1.25

    def to_dict(self):
        return asdict(self)


def rolling_mean(values, width):
    """Centred mean excluding NaNs, with shorter windows at the boundaries."""
    values = np.asarray(values, dtype=np.float64)
    if width < 1:
        raise ValueError("Window width must be positive")
    valid = np.isfinite(values)
    sums = np.concatenate(([0.0], np.cumsum(np.where(valid, values, 0.0))))
    counts = np.concatenate(([0], np.cumsum(valid)))
    index = np.arange(len(values))
    lo = np.maximum(0, index - width // 2)
    hi = np.minimum(len(values), index + (width + 1) // 2)
    number = counts[hi] - counts[lo]
    return np.divide(sums[hi] - sums[lo], number,
                     out=np.full(len(values), np.nan), where=number > 0)


def fill_short_gaps(values, limit):
    """Linearly interpolate short gaps only when observed on both sides."""
    values = np.asarray(values, dtype=np.float64)
    out = values.copy()
    known = np.flatnonzero(np.isfinite(values).all(axis=1))
    if len(known) < 2:
        return out
    for left, right in zip(known[:-1], known[1:]):
        if 1 < right - left <= limit + 1:
            fraction = np.arange(1, right - left) / (right - left)
            out[left + 1:right] = values[left] + fraction[:, None] * (values[right] - values[left])
    return out


def activity_evidence(speed, fit_mask, fps):
    """Fit two local movement regimes with deterministic unsupervised EM.

    These are NOT learned alive/dead labels: the faster regime only supplies
    positive movement evidence. Missing speeds remain neutral.
    """
    x = np.log1p(np.clip(speed, 0, 15))
    fit = x[fit_mask][::fps]
    fit = fit[np.isfinite(fit)]
    if len(fit) < 10 or np.std(fit) < 0.03:
        return np.zeros(len(speed)), {"fitted": False}
    means = np.quantile(fit, [0.2, 0.8])
    variances = np.full(2, max(float(np.var(fit)), 0.02))
    weights = np.full(2, 0.5)
    for _ in range(60):
        logp = (-0.5 * (fit[:, None] - means) ** 2 / variances
                - 0.5 * np.log(variances) + np.log(weights))
        p = np.exp(logp - np.max(logp, axis=1, keepdims=True))
        p /= np.sum(p, axis=1, keepdims=True)
        mass = p.sum(axis=0) + 1e-9
        new_means = (p * fit[:, None]).sum(axis=0) / mass
        variances = np.maximum((p * (fit[:, None] - new_means) ** 2).sum(axis=0) / mass, 0.01)
        weights = mass / mass.sum()
        change = np.max(np.abs(new_means - means))
        means = new_means
        if change < 1e-7:
            break
    order = np.argsort(means)
    means, variances, weights = means[order], variances[order], weights[order]
    likelihood = (-0.5 * (x[:, None] - means) ** 2 / variances
                  - 0.5 * np.log(variances) + np.log(weights))
    evidence = np.nan_to_num(likelihood[:, 1] - likelihood[:, 0], nan=0.0)
    return np.clip(evidence, -4, 4), {"fitted": True,
        "regime_speed_m_s": np.expm1(means).tolist(), "weights": weights.tolist()}


def team_side(value, names):
    value = normalise_name(value).replace("athletic", "atletic")
    if value in ("", "na", "none", "unknown"):
        return None
    options = [normalise_name(n).replace("athletic", "atletic") for n in names]
    scores = [1.0 if value in name or name in value else SequenceMatcher(None, value, name).ratio()
              for name in options]
    best = int(np.argmax(scores))
    return best if scores[best] >= 0.7 and abs(scores[0] - scores[1]) > 0.1 else None


ATTACK_CODES = {"buildup", "progression", "creatingchances", "attackingtransition",
                "longball", "ballinfinalthird", "ballinthebox", "playersinthebox"}
DEFENCE_CODES = {"defendinginattackingthird", "defendinginmiddlethird",
                 "defendingindefensivethird", "defensivetransition"}


def tagging_evidence(events, names, n, fps, ball_speed, collective_speed):
    """Combine correlated tags by their maximum rather than counting them repeatedly."""
    support = np.zeros((n, 2))
    live = np.zeros(n)
    dead = np.zeros(n)
    restart_live = np.zeros(n)
    unknown_teams = 0
    for event in events:
        start, stop, code = event["start"], event["stop"], event["code"]
        if stop <= start:
            continue
        side = team_side(event["tags"].get("team"), names)
        if side is None and normalise_name(event["tags"].get("team")) not in ("", "na"):
            unknown_teams += 1
        if code in ATTACK_CODES | DEFENCE_CODES:
            edge = np.minimum(np.arange(stop - start) + 1, np.arange(stop - start, 0, -1))
            taper = np.minimum(1, edge / max(1, fps / 2))
            strength = 1.0 if code in {"buildup", "progression", "creatingchances"} else 0.6
            if "transition" in code:
                strength = 1.3
            if side is not None:
                owner = 1 - side if code in DEFENCE_CODES else side
                support[start:stop, owner] = np.maximum(support[start:stop, owner], strength * taper)
            live[start:stop] = np.maximum(live[start:stop], 0.7 * taper)
        elif code == "setpieces":
            # Detect movement onset within the automatically marked restart
            # window, allowing a short delay; no manually specified timestamps.
            lag = max(1, fps // 2)
            idx = np.arange(start, stop)
            before = ball_speed[np.maximum(0, idx - lag)]
            after = ball_speed[np.minimum(n - 1, idx + lag)]
            onset = np.clip(after - before, 0, 12)
            good = np.isfinite(before) & np.isfinite(after) & (before < 2.5) & (after > 3)
            score = np.where(good, onset, 0.0)
            if np.max(score) > 1:
                anchor = int(idx[np.argmax(score)])
                strength = 2.5
            else:
                # Use only a weak interval prior when the ball is not observed.
                motion_change = (collective_speed[np.minimum(n - 1, idx + lag)]
                                 - collective_speed[np.maximum(0, idx - lag)])
                motion_change = np.nan_to_num(motion_change)
                anchor = int(idx[np.argmax(motion_change)]) if np.max(motion_change) > 0.3 else (start + stop) // 2
                strength = 0.8
            dead[start:anchor] = np.maximum(dead[start:anchor], strength)
            restart_live[anchor:stop] = np.maximum(restart_live[anchor:stop], strength * 0.6)
            if side is not None:
                support[anchor:stop, side] = np.maximum(support[anchor:stop, side], 1.0)
    return support[:, 0] - support[:, 1], live - dead + restart_live, unknown_teams


PROFILES = ("incumbent", "reliable", "responsive", "controlled")
TRACE_FIELDS = ("geometry", "tag_team", "team", "activity", "moving", "missing",
                "stationary", "slow", "outside", "status_tags", "alive",
                "ball_observed", "ball_rejected")


def profile_config(profile):
    """Explicit experimental defaults; caller-supplied configurations take precedence."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown model profile: {profile}")
    # Cost 12 was selected on independent development games and then evaluated
    # unchanged on two reserved matches (control-switch-cost-002).
    return Config(team_switch_cost=12.0) if profile in ("responsive", "controlled") else Config()


def controlled_geometry(distance, player_speed, ball_speed, valid, dead_like, fps, config):
    """Compute bounded temporal-control geometry evidence within a single period block.

    Distinguishes sustained close ball control (including contested receptions)
    from brief opponent proximity, fast passing flight, and weak long-range
    proximity, while retaining last verified close control across pass flight.
    """
    length = len(distance)
    if length == 0:
        return np.zeros(0, dtype=np.float64)
    d_h = np.nan_to_num(distance[:, 0], nan=1e6)
    d_a = np.nan_to_num(distance[:, 1], nan=1e6)
    min_d = np.minimum(d_h, d_a)
    near = np.nan_to_num(np.exp(-min_d / config.control_radius_m))
    base = np.where(
        valid,
        np.clip((d_a - d_h) / config.distance_margin_m, -2.5, 2.5) * near,
        0.0,
    )
    base *= 1.0 - 0.7 * np.nan_to_num(np.clip((ball_speed - 10.0) / 15.0, 0.0, 1.0))

    eff_h = np.maximum(0.0, d_h - 0.6)
    eff_a = np.maximum(0.0, d_a - 0.6)
    raw_margin = np.clip((eff_a - eff_h) / 1.0, -2.5, 2.5)

    close_reach = np.exp(-((min_d / 1.8) ** 2))
    v_h = np.nan_to_num(player_speed[:, 0], nan=3.0)
    v_a = np.nan_to_num(player_speed[:, 1], nan=3.0)
    v_near = np.where(d_h < d_a, v_h, np.where(d_a < d_h, v_a, 0.5 * (v_h + v_a)))
    b_speed = np.nan_to_num(ball_speed, nan=0.0)
    excess_speed = np.maximum(0.0, b_speed - np.clip(v_near, 0.0, 12.0))
    speed_gate = 1.0 - 0.85 * np.clip((excess_speed - 5.0) / 10.0, 0.0, 1.0)
    w_close = np.where(valid, close_reach * speed_gate, 0.0)
    g_inst = np.where(valid, raw_margin * w_close, 0.0)

    w = max(3, round(0.28 * fps))
    unit = np.sign(g_inst) * np.minimum(1.0, np.abs(g_inst) / 0.5)
    csum = np.concatenate(([0.0], np.cumsum(unit)))
    idx = np.arange(length)
    b_lo, b_hi = np.maximum(0, idx - w + 1), idx + 1
    c_lo, c_hi = np.maximum(0, idx - w // 2), np.minimum(length, idx + (w + 1) // 2)
    f_lo, f_hi = idx, np.minimum(length, idx + w)
    mean_b = (csum[b_hi] - csum[b_lo]) / np.maximum(1, b_hi - b_lo)
    mean_c = (csum[c_hi] - csum[c_lo]) / np.maximum(1, c_hi - c_lo)
    mean_f = (csum[f_hi] - csum[f_lo]) / np.maximum(1, f_hi - f_lo)
    sign = np.sign(g_inst)
    same_dir = np.maximum(np.maximum(sign * mean_b, sign * mean_c), sign * mean_f)
    persistence = np.clip((same_dir - 0.35) / 0.40, 0.0, 1.0)
    g_close = g_inst * (0.15 + 0.85 * persistence)

    log_live = -1.0 / max(1.0, 2.5 * fps)
    log_dead = -1.0 / max(1.0, 0.35 * fps)
    active = (~dead_like) & (w_close >= 0.35) & (persistence >= 0.5) & (np.abs(g_close) >= 0.75)
    log_decay = np.where(active, 0.0, np.where(dead_like, log_dead, log_live))
    cum_log_decay = np.cumsum(log_decay)

    act_idx = np.flatnonzero(active)
    m = np.zeros(length, dtype=np.float64)
    if len(act_idx):
        act_wp = (w_close[act_idx] * persistence[act_idx]).tolist()
        act_tgt = np.clip(g_close[act_idx], -1.6, 1.6).tolist()
        act_cl = cum_log_decay[act_idx].tolist()
        act_m = [0.0] * len(act_idx)
        state = 0.0
        prev_cl = 0.0
        exp = np.exp
        for k, (wp, tgt, cl) in enumerate(zip(act_wp, act_tgt, act_cl)):
            if cl != prev_cl:
                state *= float(exp(cl - prev_cl))
                prev_cl = cl
            alpha = 0.65 * wp if state * tgt < 0.0 else 0.35 * wp
            if alpha > 1.0:
                alpha = 1.0
            state = (1.0 - alpha) * state + alpha * tgt
            act_m[k] = state
        last_pos = np.maximum.accumulate(np.where(active, idx, -1))
        has_prev = last_pos >= 0
        m_anchor = np.zeros(length, dtype=np.float64)
        m[act_idx] = act_m
        m_anchor[has_prev] = m[last_pos[has_prev]]
        cl_anchor = np.zeros(length, dtype=np.float64)
        cl_anchor[has_prev] = cum_log_decay[last_pos[has_prev]]
        m = np.where(has_prev, m_anchor * np.exp(cum_log_decay - cl_anchor), 0.0)

    r = np.clip(w_close * (0.25 + 0.75 * persistence), 0.0, 1.0)
    c_mem = np.clip(np.abs(m) / 0.6, 0.0, 1.0)
    g_loose = c_mem * (0.85 * m) + (1.0 - c_mem) * (0.5 * base)
    combined = np.clip(r * g_close + (1.0 - r) * g_loose, -2.5, 2.5)
    return np.where(valid, combined, 0.0)


def reliable_ball_mask(ball, observed, pitch, fps, max_speed=45.0):
    """Reject isolated motion-inconsistent positions without inventing ball control.

    Both adjacent steps must exceed the existing ball-speed plausibility bound,
    while the neighbouring positions must connect plausibly over two frames.
    Missing endpoints supply no evidence of a spike. At alternating implausible
    positions, abstaining from geometry is preferable to inventing an owner.
    This experimental filter never reads labels or tagging.
    """
    ball = np.asarray(ball, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    if ball.ndim != 2 or ball.shape != (len(observed), 2):
        raise ValueError("Ball positions must have shape (frames, 2)")
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(max_speed) or max_speed <= 0:
        raise ValueError("Motion plausibility requires positive finite fps and maximum speed")
    trusted = observed.copy()
    if len(ball) < 3:
        return trusted
    xy = ball * np.asarray(pitch, dtype=np.float64)
    step_speed = np.linalg.norm(np.diff(xy, axis=0), axis=1) * fps
    bridge_speed = np.linalg.norm(xy[2:] - xy[:-2], axis=1) * fps / 2
    complete = observed[:-2] & observed[1:-1] & observed[2:]
    rejected = (complete & (step_speed[:-1] > max_speed) & (step_speed[1:] > max_speed)
                & (bridge_speed <= max_speed))
    trusted[1:-1] &= ~rejected
    return trusted


def emissions(features, events, blocks, pitch, fps, config=None, *, profile="incumbent", trace=None):
    if profile not in PROFILES:
        raise ValueError(f"Unknown model profile: {profile}")
    if config is None:
        config = profile_config(profile)
    n = len(features.ball)
    if trace is not None:
        trace.clear()
        trace.update({name: np.full(n, np.nan) for name in TRACE_FIELDS})
        trace["scores"] = np.full((n, 3), np.nan)
    fit_mask = np.zeros(n, dtype=bool)
    for _, start, stop in blocks:
        fit_mask[start:stop + 1] = True
    # Compute windows strictly within each half; never carry evidence across breaks.
    out = np.zeros((n, 3))
    activity, model_report = activity_evidence(features.mean_speed, fit_mask, fps)
    unknown = 0
    for _, start, last in blocks:
        stop = last + 1
        length = stop - start
        ball = features.ball[start:stop, :2]
        sensible = np.isfinite(ball).all(axis=1) & (np.abs(ball - 0.5) <= 0.65).all(axis=1)
        observed = sensible.copy()
        if profile == "reliable":
            sensible = reliable_ball_mask(ball, observed, pitch, fps)
        ball = np.where(sensible[:, None], ball, np.nan)
        xy = fill_short_gaps(ball, round(config.short_gap_s * fps)) * np.asarray(pitch)
        lag = max(1, fps // 8)
        index = np.arange(length)
        lo, hi = np.maximum(0, index - lag), np.minimum(length - 1, index + lag)
        ball_speed = np.linalg.norm(xy[hi] - xy[lo], axis=1) * fps / np.maximum(1, hi - lo)
        ball_speed[ball_speed > 45] = np.nan
        window = max(1, round(config.motion_window_s * fps))
        motion = rolling_mean(features.mean_speed[start:stop], window)
        missing = 1 - np.nan_to_num(rolling_mean(sensible.astype(float), window))
        moving = np.nan_to_num(np.clip((rolling_mean(ball_speed, max(1, fps // 4)) - 1.5) / 4, 0, 1))
        stationary = np.nan_to_num(np.clip((1.0 - rolling_mean(ball_speed, window)), 0, 1))
        slow = np.nan_to_num(np.clip((1.8 - motion) / 1.2, 0, 1))
        outside = sensible & ((ball[:, 0] < -0.005) | (ball[:, 0] > 1.005)
                              | (ball[:, 1] < -0.008) | (ball[:, 1] > 1.008))
        distance = features.distance[start:stop]
        valid = np.isfinite(distance).all(axis=1) & sensible
        if profile == "controlled":
            dead_like = outside | ((stationary * slow) > 0.5)
            geometry = controlled_geometry(
                distance, features.player_speed[start:stop], ball_speed,
                valid, dead_like, fps, config,
            )
        else:
            near = np.nan_to_num(np.exp(-np.nan_to_num(np.min(distance, axis=1), nan=1e6) / config.control_radius_m))
            geometry = np.where(valid, np.clip(np.nan_to_num(distance[:, 1] - distance[:, 0])
                                / config.distance_margin_m, -2.5, 2.5) * near, 0.0)
            # A fast loose ball does not prove that the nearest player controls it.
            geometry *= 1 - 0.7 * np.nan_to_num(np.clip((ball_speed - 10) / 15, 0, 1))
        local_events = [{**e, "start": max(0, e["start"] - start),
                         "stop": min(length, e["stop"] - start)}
                        for e in events if e["start"] < stop and e["stop"] > start]
        tags, status_tags, unknown_local = tagging_evidence(
            local_events, features.teams, length, fps, ball_speed, motion)
        unknown += unknown_local
        team = geometry + config.tag_strength * tags
        alive = (0.4 + 0.7 * activity[start:stop] + 1.8 * moving - 0.4 * missing
                 - 1.2 * stationary * slow - 3.0 * outside + status_tags)
        alive = np.clip(alive, -7, 7)
        log_alive = -np.logaddexp(0, -alive)
        log_dead = -np.logaddexp(0, alive)
        # Relative team costs: missing team evidence must not automatically favour D.
        out[start:stop, 0] = log_alive + np.minimum(team, 0)
        out[start:stop, 1] = log_alive - np.maximum(team, 0)
        out[start:stop, 2] = log_dead
        if trace is not None:
            values = (geometry, tags, team, activity[start:stop], moving, missing,
                      stationary, slow, outside, status_tags, alive, observed, observed & ~sensible)
            for name, value in zip(TRACE_FIELDS, values):
                trace[name][start:stop] = value
            trace["scores"][start:stop] = out[start:stop]
    return out, {"profile": profile, "experimental": profile != "incumbent",
                 "activity_model": model_report, "unknown_team_tags": unknown}


def decode(scores, switch_cost, min_frames=1):
    """Exact Viterbi path with a minimum dwell time and no window-induced latency.

    Each class has min_frames age states; only the last may be exited.
    The first/last state may be shorter at an observation boundary.
    """
    scores = np.asarray(scores, dtype=np.float64)
    cost = np.asarray(switch_cost, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != 3 or not np.isfinite(scores).all():
        raise ValueError("Decoder requires finite scores with shape (frames, 3)")
    if cost.shape != (3, 3) or not np.isfinite(cost).all() or np.any(cost < 0):
        raise ValueError("Invalid switching costs")
    if not isinstance(min_frames, (int, np.integer)) or not 1 <= min_frames <= 80:
        raise ValueError("Minimum duration must be between 1 and 80 frames")
    n = len(scores)
    if n == 0:
        return np.empty(0, dtype=np.int8)
    k = min_frames
    # This recurrence is sequential and its arrays contain only a few values.
    # Python float lists avoid millions of tiny NumPy calls here.
    columns = scores.T.tolist()
    penalties = cost.tolist()
    previous = [[-float("inf")] * k for _ in range(3)]
    for team in range(3):
        previous[team][-1] = columns[team][0]  # allow a shorter first state
    # Internal age transitions are deterministic. For each frame/team, store
    # only the entry predecessor (bits 0–1) and whether the state persists (bit 2).
    back = bytearray(n * 3)
    alternatives = ((1, 2), (0, 2), (0, 1))
    for frame in range(1, n):
        current = []
        for team, (left, right) in enumerate(alternatives):
            left_score = previous[left][-1] - penalties[left][team]
            right_score = previous[right][-1] - penalties[right][team]
            # Match np.argmax: ties favour the smaller team index.
            source, enter = (left, left_score) if left_score >= right_score else (right, right_score)
            emission = columns[team][frame]
            old = previous[team]
            if k > 1:
                row = [enter + emission] + [value + emission for value in old[:-1]]
                stay = old[-1] >= old[-2]
                if stay:
                    row[-1] = old[-1] + emission
            else:
                stay = old[-1] >= enter
                row = [(old[-1] if stay else enter) + emission]
            current.append(row)
            back[frame * 3 + team] = source | (4 if stay else 0)
        previous = current
    state = int(np.argmax(previous))
    team, age = divmod(state, k)
    path = np.empty(n, dtype=np.int8)
    for frame in range(n - 1, -1, -1):
        path[frame] = team
        code = back[frame * 3 + team]
        if age == k - 1 and code & 4:
            continue
        if age == 0:
            team, age = code & 3, k - 1
        else:
            age -= 1
    return path


def predict(features, events, blocks, pitch, fps, config=None, *, profile="incumbent", trace=None):
    if config is None:
        config = profile_config(profile)
    scores, report = emissions(features, events, blocks, pitch, fps, config, profile=profile, trace=trace)
    labels = np.full(len(features.ball), "X", dtype="<U1")
    cost = np.full((3, 3), config.stoppage_switch_cost)
    cost[0, 1] = cost[1, 0] = config.team_switch_cost
    np.fill_diagonal(cost, 0)
    alphabet = np.array(["H", "A", "D"])
    for _, start, last in blocks:
        labels[start:last + 1] = alphabet[decode(scores[start:last + 1], cost,
                                                 max(1, round(config.min_state_s * fps)))]
    return labels, report
