"""Strict local input validation and memory-efficient feature extraction.

Player trajectories are neither persisted nor exported. Error messages include
line numbers and schema information, never raw input records.
"""

from dataclasses import dataclass
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np


def normalise_name(value):
    return re.sub(r"[^a-z0-9]", "", (value or "").casefold())


@dataclass
class Features:
    ball: np.ndarray
    distance: np.ndarray
    player_speed: np.ndarray
    mean_speed: np.ndarray
    fast_fraction: np.ndarray
    player_count: np.ndarray
    teams: tuple


def read_metadata(path):
    root = ET.parse(path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2 or any(not t.get("id") for t in teams):
        raise ValueError("ATD metadata must contain exactly two teams with IDs")
    ids = [t.get("id") for t in teams]
    if ids[0] == ids[1]:
        raise ValueError("ATD team IDs are not unique")
    names = tuple(t.findtext("Name") or t.get("id") for t in teams)
    slots = list(root.iter("Player"))
    if not slots or any(p.get("teamId") not in ids for p in slots):
        raise ValueError("ATD track slot has no known team assignment")
    return np.array([ids.index(p.get("teamId")) for p in slots]), names


def _tracking_block(rows, first, slot_team, scale, output):
    """Process a bounded NumPy data block without persisting trajectories."""
    n_rows = len(rows)
    n_slots = len(slot_team)
    # loadtxt strictly parses complete numeric fields. Unlike fromstring,
    # it needs no sentinel to detect silently ignored trailing text.
    try:
        values = np.loadtxt(rows, dtype=np.float64, delimiter=",", comments=None, ndmin=2)
    except ValueError:
        raise ValueError(f"ATD lines {first + 1}–{first + n_rows}: invalid number") from None
    if values.shape != (n_rows, (n_slots + 1) * 3):
        raise ValueError(f"ATD lines {first + 1}–{first + n_rows}: invalid number")
    table = values.reshape(n_rows, n_slots + 1, 3)
    invalid = np.isinf(table).any(axis=(1, 2))
    if invalid.any():
        row = first + int(np.flatnonzero(invalid)[0]) + 1
        raise ValueError(f"ATD line {row}: infinite coordinate")
    players, points = table[:, :-1], table[:, -1]
    where = slice(first, first + n_rows)
    visible = np.isfinite(players[:, :, :2]).all(axis=2)
    output.ball[where] = points
    output.player_count[where] = visible.sum(axis=1)
    speeds = players[:, :, 2]
    valid_speed = visible & np.isfinite(speeds) & (speeds >= 0) & (speeds <= 15)
    counts = valid_speed.sum(axis=1)
    # Group rows with the same number of visible speed values and sum exactly
    # the same compact values as the original per-row calculation. This preserves
    # the reduction order instead of mixing missing values with zeros and
    # introducing unnecessary rounding differences.
    for count in np.unique(counts):
        if count:
            selected = counts == count
            compact = speeds[selected][valid_speed[selected]].reshape(-1, count)
            indices = first + np.flatnonzero(selected)
            output.mean_speed[indices] = compact.mean(axis=1)
            output.fast_fraction[indices] = (compact > 2.0).mean(axis=1)
    delta = (players[:, :, :2] - points[:, None, :2]) * scale
    squared_distance = np.sum(delta * delta, axis=2)
    ball_visible = np.isfinite(points[:, :2]).all(axis=1)
    for team in (0, 1):
        candidates = visible & (slot_team == team)[None, :]
        valid = candidates.any(axis=1) & ball_visible
        masked = np.where(candidates, squared_distance, np.inf)
        nearest = np.argmin(masked, axis=1)
        minimum = masked[np.arange(n_rows), nearest]
        # Extremely large finite coordinates can overflow the distance.
        # In that case, preserve the original first-visible-candidate rule.
        overflow = valid & np.isinf(minimum)
        nearest[overflow] = np.argmax(candidates[overflow], axis=1)
        indices = np.flatnonzero(valid)
        output.distance[first + indices, team] = np.sqrt(minimum[indices])
        output.player_speed[first + indices, team] = speeds[indices, nearest[indices]]


def read_tracking(meta_path, raw_path, n_frames, pitch, block_size=4096):
    """Stream the input once; every frame must appear exactly once and in order.

    The block size affects memory usage and speed only, not the features or
    inference parameters.
    """
    if not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError("Block size must be a positive integer")
    slot_team, names = read_metadata(meta_path)
    output = Features(np.full((n_frames, 3), np.nan), np.full((n_frames, 2), np.nan),
                      np.full((n_frames, 2), np.nan), np.full(n_frames, np.nan),
                      np.full(n_frames, np.nan), np.zeros(n_frames, dtype=np.int16), names)
    scale = np.asarray(pitch, dtype=np.float64)
    rows = []
    count = 0
    with Path(raw_path).open(encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle):
            if index >= n_frames:
                raise ValueError("ATD contains more frames than expected")
            parts = line.strip().split(":")
            if len(parts) != 3:
                raise ValueError(f"ATD line {index + 1}: expected frame:players:ball")
            try:
                frame_id = int(parts[0])
            except ValueError:
                raise ValueError(f"ATD line {index + 1}: invalid frame ID") from None
            if frame_id != index + 1:
                raise ValueError(f"ATD line {index + 1}: frame IDs must be consecutive from 1")
            players = parts[1].rstrip(";").replace(";", ",")
            point = parts[2].rstrip(";")
            if players.count(",") != len(slot_team) * 3 - 1 or point.count(",") != 2:
                raise ValueError(f"ATD line {index + 1}: unexpected number of tracks/coordinates")
            rows.append(players + "," + point)
            count += 1
            if len(rows) == block_size:
                _tracking_block(rows, count - len(rows), slot_team, scale, output)
                rows.clear()
    if count != n_frames:
        raise ValueError(f"ATD contains {count} frames; expected {n_frames}")
    if rows:
        _tracking_block(rows, count - len(rows), slot_team, scale, output)
    return output


def read_events(path, n_frames, fps):
    """Convert SportsCode to half-open frame intervals without hard-coded match events."""
    events = []
    for instance in ET.parse(path).iter("instance"):
        try:
            start = float(instance.findtext("start"))
            stop = float(instance.findtext("end"))
        except (ValueError, TypeError):
            raise ValueError("Tagging contains invalid timestamps") from None
        if not np.isfinite([start, stop]).all() or start < 0 or stop < start:
            raise ValueError("Tagging contains an invalid time interval")
        if start * fps >= n_frames or stop * fps > n_frames + 1:
            raise ValueError("Tagging falls outside the tactical timeline")
        tags = {normalise_name(label.findtext("group")): label.findtext("text") or ""
                for label in instance.findall("label")}
        events.append({"start": round(start * fps), "stop": min(n_frames, round(stop * fps)),
                       "code": normalise_name(instance.findtext("code")), "tags": tags})
    return events
