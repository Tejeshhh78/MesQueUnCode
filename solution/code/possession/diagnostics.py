"""Aggregate local model evidence without exposing frame-level match records.

The JSON report contains aggregate statistics only. Detailed arrays are written
only after an explicit request and a Git-ignore check on the resolved target.
Evidence scores and score margins are not calibrated probabilities.
"""

import json
from pathlib import Path
import subprocess

import numpy as np

from evaluation import possession_metrics as pm


ROOT = Path(__file__).resolve().parents[3]
TRACE_FEATURES = (
    "geometry", "tag_team", "team", "activity", "moving", "missing",
    "stationary", "slow", "outside", "status_tags", "alive",
)
OPTIONAL_TRACE_FEATURES = ("ball_observed", "ball_rejected")
LABELS = ("H", "A", "D", "X")
DURATION_BINS_SECONDS = (0, 2, 5, 15, 1000000)


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


def _label_codes(values, n):
    """Convert H/A/D/X strings or 0..3 integer codes to validated integer codes."""
    array = np.asarray(values)
    if array.shape != (n,):
        raise ValueError("Reference labels must match the prediction length")
    if np.issubdtype(array.dtype, np.integer):
        if np.any((array < 0) | (array >= len(LABELS))):
            raise ValueError("Integer reference codes must lie in 0..3")
        return array.astype(np.int8)
    if not np.isin(array, LABELS).all():
        raise ValueError("Reference labels must contain only H, A, D or X")
    return pm.label_codes(array)


def _greedy_match_pairs(gt_spells, pred_spells, threshold):
    """Return greedy one-to-one matched pairs (gt_index, pred_index, iou) at threshold."""
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)
    matched_gt = set()
    matched_pred = set()
    pairs = []
    for iou, gt_idx, pred_idx in overlaps:
        if iou < threshold:
            break
        if gt_idx in matched_gt or pred_idx in matched_pred:
            continue
        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)
        pairs.append((gt_idx, pred_idx, float(iou)))
    return pairs, matched_gt, matched_pred


def _interval_prf(n_matched, n_pred, n_gt):
    precision = n_matched / float(n_pred) if n_pred else 0.0
    recall = n_matched / float(n_gt) if n_gt else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n_gt": int(n_gt),
        "n_pred": int(n_pred),
        "n_matched": int(n_matched),
        "n_spurious": int(n_pred - n_matched),
        "n_missed": int(n_gt - n_matched),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _period_transition_summary(gt_trans_by_period, pred_trans_by_period, tol, fps):
    """Aggregate official to-typed transition matching without crossing metadata periods."""
    pairs, missed, spurious = [], [], []
    ambiguous = 0
    n_gt = 0
    n_pred = 0
    for gt_trans, pred_trans in zip(gt_trans_by_period, pred_trans_by_period):
        n_gt += len(gt_trans)
        n_pred += len(pred_trans)
        period_scores = pm.transition_scores(gt_trans, pred_trans, tol, typed="to")
        pairs.extend(period_scores["pairs"])
        missed.extend(period_scores["missed"])
        spurious.extend(period_scores["spurious"])
        ambiguous += int(period_scores["n_ambiguous"])
    summary = _interval_prf(len(pairs), n_pred, n_gt)
    latencies = np.asarray([pred[0] - gt[0] for gt, pred in pairs], dtype=np.float64)
    summary.update({
        "n_ambiguous": int(ambiguous),
        "latency_frames": _statistics(latencies),
        "abs_latency_frames": _statistics(np.abs(latencies)),
        "latency_s": _statistics(latencies / fps),
    })
    return summary


def reference_diagnostics(reference, labels, trace, blocks, fps=25):
    """Return aggregate reference-vs-prediction diagnostics without frame timelines.

    Implements the predeclared Stage 2 diagnostic definitions:
    - unchanged official greedy same-team IoU matching within native metadata
      periods and known-reference runs;
    - likely splits, likely merges, and wrong-team overlap counts;
    - recall and failure counts by reference possession duration bins;
    - missing-ball, suppressed-correct-geometry, and selected-wrong-geometry frame
      counts (descriptive associations, not proof of decoder causality);
    - signed start/end boundary errors on IoU-0.5 matched possessions;
    - stoppage frame/interval counts and transition/restart errors at 12 frames.
    """
    arrays, pred_labels, ranges, covered, fps = _prepare(trace, labels, blocks, fps)
    ref_codes = _label_codes(reference, len(pred_labels))
    pred_codes = pm.label_codes(pred_labels)
    if np.any(ref_codes[~covered] != pm.X):
        raise ValueError("Reference labels outside declared period blocks must be X")

    period_runs = []
    runs = []
    for first, stop in ranges:
        sub_runs = [(first + a, first + b) for a, b in pm.valid_runs(ref_codes[first:stop])]
        period_runs.append(sub_runs)
        runs.extend(sub_runs)

    filled_pred = pm.fill_abstentions(pred_codes, runs)
    gt_spells = pm.possession_spells(ref_codes, runs)
    pred_spells = pm.possession_spells(filled_pred, runs)
    spell_summary = pm.spell_scores(ref_codes, filled_pred, runs)
    pairs_05, matched_gt_05, matched_pred_05 = _greedy_match_pairs(
        gt_spells, pred_spells, pm.SPELL_IOU_HEADLINE
    )

    gt_likely_split = np.zeros(len(gt_spells), dtype=bool)
    for i, (gt_start, gt_stop, gt_code) in enumerate(gt_spells):
        gt_len = float(gt_stop - gt_start)
        qualifying = 0
        for pred_start, pred_stop, pred_code in pred_spells:
            if pred_code != gt_code or pred_stop <= gt_start or gt_stop <= pred_start:
                continue
            overlap = min(gt_stop, pred_stop) - max(gt_start, pred_start)
            if overlap / gt_len >= 0.10:
                qualifying += 1
        gt_likely_split[i] = qualifying >= 2

    pred_likely_merge = np.zeros(len(pred_spells), dtype=bool)
    for j, (pred_start, pred_stop, pred_code) in enumerate(pred_spells):
        pred_len = float(pred_stop - pred_start)
        qualifying = 0
        for gt_start, gt_stop, gt_code in gt_spells:
            if gt_code != pred_code or pred_stop <= gt_start or gt_stop <= pred_start:
                continue
            overlap = min(gt_stop, pred_stop) - max(gt_start, pred_start)
            if overlap / pred_len >= 0.10:
                qualifying += 1
        pred_likely_merge[j] = qualifying >= 2

    graded = covered & (ref_codes != pm.X)
    wrong_team_mask = graded & (
        ((ref_codes == pm.H) & (pred_codes == pm.A))
        | ((ref_codes == pm.A) & (pred_codes == pm.H))
    )
    gt_has_wrong_team = np.zeros(len(gt_spells), dtype=bool)
    gt_majority_wrong_team = np.zeros(len(gt_spells), dtype=bool)
    for i, (gt_start, gt_stop, gt_code) in enumerate(gt_spells):
        opp_code = pm.A if gt_code == pm.H else pm.H
        opp_frames = int(np.sum(pred_codes[gt_start:gt_stop] == opp_code))
        gt_has_wrong_team[i] = opp_frames > 0
        gt_majority_wrong_team[i] = opp_frames > 0.5 * (gt_stop - gt_start)

    duration_bins = {}
    for low, high in zip(DURATION_BINS_SECONDS[:-1], DURATION_BINS_SECONDS[1:]):
        key = f"{low}_{high}s"
        indices = [
            i for i, (gt_start, gt_stop, _) in enumerate(gt_spells)
            if low <= (gt_stop - gt_start) / fps < high
        ]
        n_bin = len(indices)
        n_matched_bin = sum(1 for i in indices if i in matched_gt_05)
        duration_bins[key] = {
            "min_s": float(low),
            "max_s": float(high),
            "reference_count": int(n_bin),
            "matched_iou_05_count": int(n_matched_bin),
            "missed_iou_05_count": int(n_bin - n_matched_bin),
            "recall_iou_05": float(n_matched_bin / n_bin) if n_bin else None,
            "likely_split_count": int(sum(1 for i in indices if gt_likely_split[i])),
            "wrong_team_overlap_count": int(sum(1 for i in indices if gt_has_wrong_team[i])),
            "majority_wrong_team_count": int(sum(1 for i in indices if gt_majority_wrong_team[i])),
        }

    geometry = arrays["geometry"]
    finite_geom = np.isfinite(geometry)
    geom_supports_ref = graded & finite_geom & (
        ((ref_codes == pm.H) & (geometry > 0))
        | ((ref_codes == pm.A) & (geometry < 0))
    )
    geom_opposes_ref = graded & finite_geom & (
        ((ref_codes == pm.H) & (geometry < 0))
        | ((ref_codes == pm.A) & (geometry > 0))
    )
    suppressed_correct_geometry = geom_supports_ref & (pred_codes != ref_codes)
    selected_wrong_geometry = geom_opposes_ref & wrong_team_mask
    if "ball_observed" in arrays:
        missing_ball = graded & np.isfinite(arrays["ball_observed"]) & (arrays["ball_observed"] <= 0.5)
    else:
        missing_ball = graded & np.isfinite(arrays["missing"]) & (arrays["missing"] > 0.5)

    start_errors = np.asarray(
        [pred_spells[p][0] - gt_spells[g][0] for g, p, _ in pairs_05], dtype=np.float64
    )
    end_errors = np.asarray(
        [pred_spells[p][1] - gt_spells[g][1] for g, p, _ in pairs_05], dtype=np.float64
    )

    gt_dead_spells = [(a, b, k) for a, b, k in pm.segments(ref_codes, runs) if k == pm.D]
    pred_dead_spells = [(a, b, k) for a, b, k in pm.segments(filled_pred, runs) if k == pm.D]
    dead_pairs_05, _, _ = _greedy_match_pairs(
        gt_dead_spells, pred_dead_spells, pm.SPELL_IOU_HEADLINE
    )

    gt_trans_periods = [pm.transitions(ref_codes, sub_runs) for sub_runs in period_runs]
    pred_trans_periods = [pm.transitions(filled_pred, sub_runs) for sub_runs in period_runs]
    gt_restarts_periods = [
        [t for t in trans if t[1] == pm.D and t[2] in pm.POSSESSION_LABELS]
        for trans in gt_trans_periods
    ]
    pred_restarts_periods = [
        [t for t in trans if t[1] == pm.D and t[2] in pm.POSSESSION_LABELS]
        for trans in pred_trans_periods
    ]

    return {
        "schema_version": 1,
        "caveat": (
            "Categories overlap; they do not partition FP/FN or establish physical causes. "
            "Raw timelines remain local."
        ),
        "matching": {
            "spell_f1_mean": float(spell_summary["mean_f1"]),
            "n_gt_spells": int(spell_summary["n_gt"]),
            "n_pred_spells": int(spell_summary["n_pred"]),
            "iou_05": spell_summary["per_iou"][pm.SPELL_IOU_HEADLINE],
            "per_iou": {
                f"{iou:.1f}": values for iou, values in spell_summary["per_iou"].items()
            },
        },
        "structural_errors": {
            "likely_split_count": int(gt_likely_split.sum()),
            "unmatched_likely_split_count": int(
                sum(1 for i in range(len(gt_spells)) if gt_likely_split[i] and i not in matched_gt_05)
            ),
            "likely_merge_count": int(pred_likely_merge.sum()),
            "unmatched_likely_merge_count": int(
                sum(1 for j in range(len(pred_spells)) if pred_likely_merge[j] and j not in matched_pred_05)
            ),
            "wrong_team_overlap_frames": int(wrong_team_mask.sum()),
            "wrong_team_overlap_reference_spells": int(gt_has_wrong_team.sum()),
            "majority_wrong_team_reference_spells": int(gt_majority_wrong_team.sum()),
        },
        "duration_bins_seconds": duration_bins,
        "evidence_associations": {
            "graded_frames": int(graded.sum()),
            "missing_ball_frames": int(missing_ball.sum()),
            "missing_ball_error_frames": int(np.sum(missing_ball & (pred_codes != ref_codes))),
            "suppressed_correct_geometry_frames": int(suppressed_correct_geometry.sum()),
            "selected_wrong_geometry_frames": int(selected_wrong_geometry.sum()),
        },
        "boundary_error_iou_05": {
            "matched_spell_count": int(len(pairs_05)),
            "start_error_frames": _statistics(start_errors),
            "end_error_frames": _statistics(end_errors),
            "abs_start_error_frames": _statistics(np.abs(start_errors)),
            "abs_end_error_frames": _statistics(np.abs(end_errors)),
            "start_error_s": _statistics(start_errors / fps),
            "end_error_s": _statistics(end_errors / fps),
        },
        "stoppage": {
            "missed_dead_frames": int(np.sum(graded & (ref_codes == pm.D) & (pred_codes != pm.D))),
            "false_dead_frames": int(np.sum(graded & (ref_codes != pm.D) & (pred_codes == pm.D))),
            "dead_intervals_iou_05": _interval_prf(
                len(dead_pairs_05), len(pred_dead_spells), len(gt_dead_spells)
            ),
            "transitions_12": _period_transition_summary(
                gt_trans_periods, pred_trans_periods, 12, fps
            ),
            "restart_transitions_12": _period_transition_summary(
                gt_restarts_periods, pred_restarts_periods, 12, fps
            ),
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
