"""Spell clean-up after decoding, the submission writer, and spell statistics.

The decoder already prices every change of state, so this is a safety net with football meaning:

  * a "possession" shorter than min_spell_seconds is not a possession (CHALLENGE.md: a touch
    without control does not change possession) -- it is absorbed by its neighbours;
  * a dead-ball blip shorter than min_dead_seconds between two stretches of the same team is a
    tracking artefact, not a stoppage -- because H D H is scored as *two* possessions, leaving it
    in costs a whole spell.

Every operation is local and deterministic, and runs inside one half at a time so nothing ever
crosses halftime.
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

from solution.possession_model import A, D, H, STATE_LABELS

Segment = Tuple[int, int, int]  # (start, stop, state), half-open


def segments(states: np.ndarray) -> List[Segment]:
    """Constant-state runs of one span."""
    if len(states) == 0:
        return []
    breaks = np.flatnonzero(states[1:] != states[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(states)]))
    return [(int(bounds[i]), int(bounds[i + 1]), int(states[bounds[i]]))
            for i in range(len(bounds) - 1)]


def _absorb_short_spells(states: np.ndarray, min_len: int) -> bool:
    """Relabel the shortest too-short possession spell once. True if something changed."""
    segs = segments(states)
    short = [(stop - start, k) for k, (start, stop, state) in enumerate(segs)
             if state in (H, A) and stop - start < min_len]
    if not short:
        return False
    _, k = min(short)
    start, stop, state = segs[k]
    left = segs[k - 1][2] if k > 0 else None
    right = segs[k + 1][2] if k + 1 < len(segs) else None
    if left is None and right is None:
        return False
    # Adjacent runs always differ, so a team neighbour is necessarily the other team. With one on
    # either side the spell is a blip inside that team's possession; with one on one side only it
    # is a tail of that possession; with none it sits between stoppages and is not a possession.
    teams = [s for s in (left, right) if s in (H, A)]
    states[start:stop] = teams[0] if teams else D
    return True


def _bridge_short_dead(states: np.ndarray, min_len: int) -> None:
    """Fill D runs shorter than min_len that sit between two runs of the same team."""
    segs = segments(states)
    for k in range(1, len(segs) - 1):
        start, stop, state = segs[k]
        if state == D and stop - start < min_len and segs[k - 1][2] == segs[k + 1][2] \
                and segs[k - 1][2] in (H, A):
            states[start:stop] = segs[k - 1][2]


def clean_span(states: np.ndarray, min_spell: int, min_dead: int) -> np.ndarray:
    out = states.copy()
    if min_dead > 0:
        _bridge_short_dead(out, min_dead)
    if min_spell > 0:
        while _absorb_short_spells(out, min_spell):
            pass
    if min_dead > 0:
        _bridge_short_dead(out, min_dead)
    return out


def clean(states: np.ndarray, halves: Sequence[Tuple[int, int, int]], min_spell: int,
          min_dead: int) -> np.ndarray:
    out = states.copy()
    for _, first, last in halves:
        out[first:last + 1] = clean_span(states[first:last + 1], min_spell, min_dead)
    return out


def to_labels(states: np.ndarray) -> np.ndarray:
    """H/A/D for decoded frames, X everywhere else (ungraded: before, halftime, after)."""
    labels = np.full(len(states), "X", dtype="<U1")
    decoded = states >= 0
    labels[decoded] = STATE_LABELS[states[decoded]]
    return labels


def write_submission(path: str, labels: Sequence[str]) -> str:
    """frame,label CSV with '\\n' line endings, byte-identical for identical labels."""
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))
    return path


def spell_stats(labels: np.ndarray, halves: Sequence[Tuple[int, int, int]],
                fps: int) -> Dict[str, float]:
    """Possession count, mean duration, stoppages and label shares over the graded frames."""
    spells: List[int] = []
    stoppages = 0
    graded = 0
    counts = {"H": 0, "A": 0, "D": 0, "X": 0}
    for _, first, last in halves:
        span = labels[first:last + 1]
        graded += len(span)
        for label in counts:
            counts[label] += int(np.sum(span == label))
        codes = np.array([{"H": H, "A": A, "D": D}.get(v, -1) for v in span], dtype=np.int8)
        for start, stop, state in segments(codes):
            if state in (H, A):
                spells.append(stop - start)
            elif state == D:
                stoppages += 1
    return {
        "possessions": len(spells),
        "mean_possession_s": float(np.mean(spells)) / fps if spells else 0.0,
        "stoppages": stoppages,
        "graded_frames": graded,
        "share_H": counts["H"] / float(graded) if graded else 0.0,
        "share_A": counts["A"] / float(graded) if graded else 0.0,
        "share_D": counts["D"] / float(graded) if graded else 0.0,
        "share_X": counts["X"] / float(graded) if graded else 0.0,
    }
