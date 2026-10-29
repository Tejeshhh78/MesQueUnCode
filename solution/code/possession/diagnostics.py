"""Aggregate local model evidence without exposing frame-level match records.

The JSON report contains aggregate statistics only. Detailed arrays are written
only after an explicit request and a Git-ignore check on the resolved target.
Evidence scores and score margins are not calibrated probabilities.
"""

import json
from pathlib import Path
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
TRACE_FEATURES = (
    "geometry", "tag_team", "team", "activity", "moving", "missing",
    "stationary", "slow", "outside", "status_tags", "alive",
)
OPTIONAL_TRACE_FEATURES = ("ball_observed", "ball_rejected")
LABELS = ("H", "A", "D", "X")


def _prepare(trace, labels, blocks, fps):
    """Validate shapes and construct the analysis mask without returning raw data."""
    values = np.asarray(labels)
    if values.ndim != 1 or not np.isin(values, LABELS).all():
        raise ValueError("Diagnostics require a one-dimensional H/A/D/X label array")
    if not np.isscalar(fps):
        raise ValueError("Diagnostics frame rate must be finite and positive")
    try:
        fps = float(fps)
    except (TypeError, ValueError):
        raise ValueError("Diagnostics frame rate must be finite and positive") from None
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Diagnostics frame rate must be finite and positive")
    n = len(values)
    covered = np.zeros(n, dtype=bool)
    ranges = []
    for block in blocks:
        if len(block) != 3:
            raise ValueError("Diagnostics blocks must contain a name, first frame and last frame")
        _, first, last = block
        if not isinstance(first, (int, np.integer)) or not isinstance(last, (int, np.integer)):
            raise ValueError("Diagnostics block boundaries must be integers")
        if first < 0 or last < first or last >= n:
            raise ValueError("Diagnostics block lies outside the label array")
        if covered[first:last + 1].any():
            raise ValueError("Diagnostics blocks must not overlap")
        covered[first:last + 1] = True
        ranges.append((int(first), int(last) + 1))
    arrays = {}
    optional = tuple(name for name in OPTIONAL_TRACE_FEATURES if name in trace)
    for name in TRACE_FEATURES + optional + ("scores",):
        if name not in trace:
            raise ValueError(f"Missing diagnostic evidence field: {name}")
        try:
            array = np.asarray(trace[name], dtype=np.float64)
        except (TypeError, ValueError):
            raise ValueError(f"Diagnostic evidence field is not numeric: {name}") from None
        expected = (n, 3) if name == "scores" else (n,)
        if array.shape != expected:
            raise ValueError(f"Invalid diagnostic evidence shape for {name}; expected {expected}")
        arrays[name] = array
    return arrays, values.astype("<U1"), ranges, covered, fps


def _statistics(values):
    """Describe finite values; represent unavailable statistics with JSON null."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    result = {
        "finite_count": int(finite.size),
        "nonfinite_count": int(values.size - finite.size),
        "finite_fraction": float(finite.size / values.size) if values.size else None,
        "mean": None, "minimum": None, "maximum": None,
        "p05": None, "p50": None, "p95": None,
    }
    if finite.size:
        with np.errstate(over="ignore", invalid="ignore"):
            quantiles = np.quantile(finite, [0.05, 0.5, 0.95])
            computed = {"mean": np.mean(finite), "minimum": np.min(finite),
                        "maximum": np.max(finite), "p05": quantiles[0],
                        "p50": quantiles[1], "p95": quantiles[2]}
        result.update({key: float(value) if np.isfinite(value) else None
                       for key, value in computed.items()})
    return result


def aggregate_diagnostics(trace, labels, blocks, fps):
    """Return JSON-safe aggregates, without frame indices or interval timestamps.

    Blocks use the same inclusive boundaries as ``predict``. Prediction runs and
    state changes are counted separately inside each block. X is kept explicit;
    it is not forward-filled as it is in some official temporal metrics.
    """
    arrays, labels, ranges, covered, fps = _prepare(trace, labels, blocks, fps)
    count = int(covered.sum())
    label_counts = {label: int(np.sum(labels[covered] == label)) for label in LABELS}
    run_counts = {label: 0 for label in LABELS}
    run_durations = {label: [] for label in LABELS}
    transitions = {f"{left}->{right}": 0 for left in LABELS for right in LABELS if left != right}
    for first, stop in ranges:
        window = labels[first:stop]
        boundaries = np.r_[0, np.flatnonzero(window[1:] != window[:-1]) + 1, len(window)]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            label = str(window[start])
            run_counts[label] += 1
            run_durations[label].append(float(end - start) / fps)
        for start in boundaries[1:-1]:
            transitions[f"{window[start - 1]}->{window[start]}"] += 1
    feature_statistics = {}
    for name in arrays:
        if name == "scores":
            continue
        values = arrays[name]
        statistics = _statistics(values[covered])
        statistics["mean_by_predicted_label"] = {
            label: _statistics(values[covered & (labels == label)])["mean"] for label in LABELS
        }
        feature_statistics[name] = statistics
    scores = arrays["scores"]
    complete_scores = covered & np.isfinite(scores).all(axis=1)
    sorted_scores = np.sort(scores[complete_scores], axis=1)
    with np.errstate(over="ignore", invalid="ignore"):
        margins = sorted_scores[:, -1] - sorted_scores[:, -2]
    local_shortfall = []
    local_winner_counts = {label: 0 for label in LABELS[:3]}
    if complete_scores.any():
        winners = np.argmax(scores[complete_scores], axis=1)
        local_winner_counts = {label: int(np.sum(winners == code))
                               for code, label in enumerate(LABELS[:3])}
    for code, label in enumerate(LABELS[:3]):
        selected = complete_scores & (labels == label)
        values = scores[selected]
        if len(values):
            with np.errstate(over="ignore", invalid="ignore"):
                local_shortfall.extend((values.max(axis=1) - values[:, code]).tolist())
    local_shortfall = np.asarray(local_shortfall, dtype=np.float64)
    finite_team = covered & np.isfinite(arrays["team"])
    opposing_team = (((labels == "H") & (arrays["team"] < 0))
                     | ((labels == "A") & (arrays["team"] > 0))) & finite_team
    geometry, tags = arrays["geometry"], arrays["tag_team"]
    opposite_signs = ((geometry < 0) & (tags > 0)) | ((geometry > 0) & (tags < 0))
    conflicting = covered & np.isfinite(geometry) & np.isfinite(tags) & opposite_signs
    return {
        "schema_version": 1,
        "interpretation": {
            "score_semantics": "Unnormalised evidence scores and margins; not calibrated probabilities.",
            "scope": "Aggregates inside supplied blocks; no frame records, indices or interval timestamps.",
            "state_changes": "Only adjacent labels inside the same block; X remains explicit.",
            "overrides": "A temporally selected state need not maximise its isolated frame score; this is not an error count.",
        },
        "scope": {"total_frames": len(labels), "analysed_frames": count,
                  "excluded_frames": len(labels) - count, "block_count": len(ranges),
                  "fps": fps, "analysed_duration_s": count / fps},
        "predictions": {
            "label_counts": label_counts,
            "label_fractions": {label: value / count if count else None for label, value in label_counts.items()},
            "state_run_counts": run_counts,
            "possession_run_count": run_counts["H"] + run_counts["A"],
            "run_duration_s": {label: _statistics(values) for label, values in run_durations.items()},
            "state_change_count": sum(transitions.values()),
            "transition_counts": transitions,
        },
        "feature_statistics": feature_statistics,
        "emission_statistics": {
            "by_label": {label: _statistics(scores[covered, code]) for code, label in enumerate(LABELS[:3])},
            "complete_score_frames": int(complete_scores.sum()),
            "best_minus_second_score": _statistics(margins),
            "exact_top_score_ties": int(np.sum(margins == 0)),
            "isolated_frame_winner_counts": local_winner_counts,
            "selected_state_score_shortfall": _statistics(local_shortfall),
            "temporal_override_frames": int(np.sum(local_shortfall > 0)),
        },
        "disagreement_counts": {
            "opposing_geometry_and_tag_frames": int(conflicting.sum()),
            "chosen_live_team_opposes_team_evidence_frames": int(opposing_team.sum()),
        },
    }


def _require_ignored_trace_target(path):
    """Reject resolved paths outside the repository or not ignored by Git."""
    try:
        relative = path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        raise ValueError("Detailed traces must be saved inside this repository at a Git-ignored path") from None
    try:
        result = subprocess.run(["git", "check-ignore", "--quiet", "--", relative.as_posix()],
                                cwd=ROOT, capture_output=True, check=False)
    except OSError:
        raise ValueError("Git is required to verify that a detailed trace stays excluded from publication") from None
    if result.returncode != 0:
        raise ValueError("Detailed trace target is not verified as Git-ignored; use solution/reports or solution/results")


def write_diagnostics(output_dir, trace, labels, blocks, fps, *, include_trace=False):
    """Write safe JSON and, on explicit request, a private Git-ignored NPZ trace.

    Returns paths to the summary and optional trace. NPZ contains only numeric
    arrays and fixed-width labels and can be loaded with ``allow_pickle=False``.
    A failed trace-safety check writes no diagnostic files.
    """
    output = Path(output_dir)
    summary_path = output / "diagnostics.json"
    trace_path = output / "evidence_trace.npz" if include_trace else None
    blocks = list(blocks)
    summary = aggregate_diagnostics(trace, labels, blocks, fps)
    arrays, labels, _, covered, fps = _prepare(trace, labels, blocks, fps)
    if trace_path is not None:
        _require_ignored_trace_target(trace_path)
    output.mkdir(parents=True, exist_ok=True)
    if trace_path is not None:
        np.savez_compressed(trace_path, **arrays, labels=labels, analysed_mask=covered, fps=np.asarray(fps))
    with summary_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return {"summary_path": str(summary_path),
            "trace_path": str(trace_path) if trace_path is not None else None}
