"""Label-independent evidence and exact temporal state inference.

Physical scales and regularisation were fixed before the public pipeline check.
They were not trained on this match.
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
    # "first": Viterbi's own tie-breaking (earliest frame). "middle": see place_tied_switches.
    tie_switch_placement: str = "first"

    def to_dict(self):
        return asdict(self)


# v1 is the method fixed before the public pipeline check; its output is unchanged.
# v2 was selected on OTHER matches only (validation/, Metrica's public sample games with labels
# derived from their own event data): control evidence concentrated within ~1.5 m of a player,
# a correspondingly lower team switching cost, a tag weight that stays robust when tags are noisy,
# and switch placement in the middle of evidence-free gaps. No data of the provided match was
# used to choose it.
PROFILES = {
    "v1": Config(),
    "v2": Config(control_radius_m=1.5, distance_margin_m=0.5, team_switch_cost=15.0,
                 tag_strength=0.6, tie_switch_placement="middle"),
}


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


def emissions(features, events, blocks, pitch, fps, config=Config()):
    n = len(features.ball)
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
    return out, {"activity_model": model_report, "unknown_team_tags": unknown}


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


def place_tied_switches(path, scores, min_frames=1):
    """Move each H<->A switch to the middle of the exactly tied run around it.

    Where the H and A scores are identical (no team evidence, typically a ball gap), every switch
    position inside that run has the same objective value, so Viterbi's choice there is only its
    tie-breaking rule, which picks the earliest frame. The middle of the run minimises the expected
    boundary error under flat evidence. Only tied frames change label, the number of switches is
    unchanged and internal states keep at least min_frames, so the result is still an exact optimum.
    """
    path = np.array(path, dtype=np.int8, copy=True)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(path)
    if n < 2:
        return path
    tied = scores[:, 0] == scores[:, 1]
    breaks = (np.flatnonzero(path[1:] != path[:-1]) + 1).tolist()
    stops = breaks + [n]
    old_start = 0
    for index, switch in enumerate(breaks):
        new_stop = stops[index + 1]
        old, new = int(path[switch - 1]), int(path[switch])
        if old == 2 or new == 2:
            old_start = switch
            continue
        low = switch
        while low > old_start and tied[low - 1]:
            low -= 1
        high = switch
        while high < new_stop and tied[high]:
            high += 1
        target = (low + high) // 2
        # Internal states must keep the minimum duration; boundary states may be shorter.
        if old_start > 0:
            target = max(target, old_start + min_frames)
        if new_stop < n:
            target = min(target, new_stop - min_frames)
        target = min(max(target, low), high)
        path[min(target, switch):max(target, switch)] = old if target > switch else new
        old_start = target
    return path


def predict(features, events, blocks, pitch, fps, config=Config()):
    scores, report = emissions(features, events, blocks, pitch, fps, config)
    labels = np.full(len(features.ball), "X", dtype="<U1")
    cost = np.full((3, 3), config.stoppage_switch_cost)
    cost[0, 1] = cost[1, 0] = config.team_switch_cost
    np.fill_diagonal(cost, 0)
    if config.tie_switch_placement not in ("first", "middle"):
        raise ValueError("tie_switch_placement must be 'first' or 'middle'")
    alphabet = np.array(["H", "A", "D"])
    minimum = max(1, round(config.min_state_s * fps))
    for _, start, last in blocks:
        block = scores[start:last + 1]
        path = decode(block, cost, minimum)
        if config.tie_switch_placement == "middle":
            path = place_tied_switches(path, block, minimum)
        labels[start:last + 1] = alphabet[path]
    return labels, report
