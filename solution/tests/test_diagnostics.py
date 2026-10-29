"""Synthetic observability checks; no competition labels or match assets are read."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from solution.code.possession.data import Features
from solution.code.possession.diagnostics import (
    ROOT, TRACE_FEATURES, aggregate_diagnostics, reference_diagnostics, write_diagnostics,
)
from solution.code.possession.model import predict


def synthetic_trace(n):
    trace = {name: np.ones(n, dtype=float) for name in TRACE_FEATURES}
    trace["scores"] = np.tile([0.0, -1.0, -2.0], (n, 1))
    return trace


class AggregateDiagnosticsTests(unittest.TestCase):
    def test_aggregate_scope_and_transitions_never_cross_blocks(self):
        labels = np.array(list("XHHADXAHHX"))
        trace = synthetic_trace(len(labels))
        trace["geometry"] = np.arange(len(labels), dtype=float)
        trace["tag_team"][2] = -1
        blocks = [(1, 1, 4), (2, 6, 8)]
        result = aggregate_diagnostics(trace, labels, blocks, 25)
        self.assertEqual(result["scope"]["analysed_frames"], 7)
        self.assertEqual(result["predictions"]["label_counts"], {"H": 4, "A": 2, "D": 1, "X": 0})
        self.assertEqual(result["predictions"]["possession_run_count"], 4)
        self.assertEqual(result["predictions"]["state_change_count"], 3)
        self.assertEqual(result["predictions"]["transition_counts"]["D->A"], 0)
        self.assertEqual(result["predictions"]["transition_counts"]["H->A"], 1)
        self.assertAlmostEqual(result["feature_statistics"]["geometry"]["mean"], 31 / 7)
        self.assertEqual(result["disagreement_counts"]["opposing_geometry_and_tag_frames"], 1)
        self.assertEqual(result["disagreement_counts"]["chosen_live_team_opposes_team_evidence_frames"], 2)
        # JSON must contain aggregates, not one record or timestamp per frame.
        encoded = json.dumps(result, allow_nan=False)
        self.assertNotIn("start_frame", encoded)
        self.assertNotIn("end_frame", encoded)
        self.assertNotIn("confidence", encoded)
        self.assertEqual(result["emission_statistics"]["temporal_override_frames"], 3)

    def test_missing_evidence_is_json_null_not_nan_or_fabricated_zero(self):
        trace = {name: np.full(4, np.nan) for name in TRACE_FEATURES}
        trace["scores"] = np.full((4, 3), np.nan)
        result = aggregate_diagnostics(trace, np.array(list("HHAA")), [(1, 0, 3)], 25)
        statistics = result["feature_statistics"]["geometry"]
        self.assertEqual(statistics["finite_count"], 0)
        self.assertEqual(statistics["finite_fraction"], 0)
        self.assertIsNone(statistics["mean"])
        self.assertIsNone(statistics["p50"])
        self.assertIsNone(result["emission_statistics"]["best_minus_second_score"]["mean"])
        self.assertEqual(result["emission_statistics"]["complete_score_frames"], 0)
        json.dumps(result, allow_nan=False)

    def test_no_analysis_blocks_produce_empty_aggregates(self):
        result = aggregate_diagnostics(synthetic_trace(3), np.array(list("XXX")), [], 25)
        self.assertEqual(result["scope"]["analysed_frames"], 0)
        self.assertEqual(result["predictions"]["state_change_count"], 0)
        self.assertIsNone(result["feature_statistics"]["geometry"]["finite_fraction"])

    def test_optional_reliability_flags_are_aggregated_without_raw_records(self):
        trace = synthetic_trace(4)
        trace["ball_observed"] = np.array([1, 1, 0, 1], dtype=float)
        trace["ball_rejected"] = np.array([0, 1, 0, 0], dtype=float)
        trace["unused_debug_data"] = np.arange(4)
        result = aggregate_diagnostics(trace, np.array(list("HHAA")), [(1, 0, 3)], 25)
        self.assertEqual(result["feature_statistics"]["ball_observed"]["mean"], 0.75)
        self.assertEqual(result["feature_statistics"]["ball_rejected"]["mean"], 0.25)
        self.assertNotIn("unused_debug_data", result["feature_statistics"])

    def test_extreme_evidence_does_not_emit_nonfinite_json_values(self):
        trace = synthetic_trace(3)
        trace["geometry"][:] = 1e308
        trace["scores"][:] = [1e308, -1e308, 0]
        result = aggregate_diagnostics(trace, np.array(list("HHH")), [(1, 0, 2)], 25)
        json.dumps(result, allow_nan=False)

    def test_scope_and_shape_errors_are_rejected(self):
        trace = synthetic_trace(3)
        with self.assertRaises(ValueError):
            aggregate_diagnostics(trace, np.array(list("HHH")), [(1, 0, 2), (2, 2, 2)], 25)
        with self.assertRaises(ValueError):
            aggregate_diagnostics(trace, np.array(list("HHH")), [(1, 0, 3)], 25)
        with self.assertRaises(ValueError):
            aggregate_diagnostics(trace, np.array(list("HHH")), [(1, 0, 2)], 0)
        with self.assertRaises(ValueError):
            aggregate_diagnostics(trace, np.array(["HOME", "H", "A"]), [(1, 0, 2)], 25)
        trace["scores"] = np.zeros((3, 2))
        with self.assertRaises(ValueError):
            aggregate_diagnostics(trace, np.array(list("HHH")), [(1, 0, 2)], 25)

    def test_writer_requires_explicit_trace_and_keeps_json_aggregate_only(self):
        labels = np.array(list("HHAA"))
        trace = synthetic_trace(4)
        with tempfile.TemporaryDirectory() as folder:
            paths = write_diagnostics(folder, trace, labels, [(1, 0, 3)], 25)
            self.assertIsNone(paths["trace_path"])
            self.assertEqual({p.name for p in Path(folder).iterdir()}, {"diagnostics.json"})
            saved = json.loads(Path(paths["summary_path"]).read_text(encoding="utf-8"))
            self.assertEqual(saved["scope"]["analysed_frames"], 4)

    def test_private_trace_refuses_a_nonignored_repository_target(self):
        target = ROOT / "diagnostic-output-not-ignored"
        with self.assertRaises(ValueError):
            write_diagnostics(target, synthetic_trace(3), np.array(list("HHH")),
                              [(1, 0, 2)], 25, include_trace=True)
        self.assertFalse(target.exists())

    def test_private_trace_refuses_an_external_directory_before_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "private-trace"
            with self.assertRaises(ValueError):
                write_diagnostics(target, synthetic_trace(3), np.array(list("HHH")),
                                  [(1, 0, 2)], 25, include_trace=True)
            self.assertFalse(target.exists())

    def test_explicit_trace_is_numeric_and_saved_only_in_an_ignored_directory(self):
        cache = ROOT / ".cache"
        cache.mkdir(exist_ok=True)
        labels = np.array(list("HHAA"))
        trace = synthetic_trace(4)
        with tempfile.TemporaryDirectory(dir=cache) as folder:
            paths = write_diagnostics(folder, trace, labels, [(1, 0, 3)], 25, include_trace=True)
            with np.load(paths["trace_path"], allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["geometry"], trace["geometry"])
                np.testing.assert_array_equal(saved["scores"], trace["scores"])
                np.testing.assert_array_equal(saved["labels"], labels)
                self.assertTrue(saved["analysed_mask"].all())


class PredictionTraceTests(unittest.TestCase):
    def test_collecting_evidence_does_not_change_predictions_in_either_profile(self):
        n = 160
        ball = np.zeros((n, 3))
        ball[:, 0] = 0.2 + np.arange(n) / 25 * 4 / 105
        ball[:, 1] = 0.5
        ball[:, 2] = 4
        features = Features(ball, np.tile([0.5, 8.0], (n, 1)), np.full((n, 2), 3.),
                            np.full(n, 3.), np.full(n, 0.6), np.full(n, 22), ("Home", "Away"))
        # Exercise the actual reliability guard, not only a clean trajectory.
        features.ball[45, :2] = [0.95, 0.95]
        blocks = [(1, 10, 69), (2, 90, 149)]
        outside = np.ones(n, dtype=bool)
        outside[10:70] = False
        outside[90:150] = False
        for profile in ("incumbent", "reliable"):
            with self.subTest(profile=profile):
                before, _ = predict(features, [], blocks, (105, 68), 25, profile=profile)
                trace = {}
                after, _ = predict(features, [], blocks, (105, 68), 25, profile=profile, trace=trace)
                np.testing.assert_array_equal(before, after)
                self.assertTrue(set(TRACE_FEATURES + ("scores",)).issubset(trace))
                for name in TRACE_FEATURES:
                    self.assertEqual(trace[name].shape, (n,))
                    self.assertTrue(np.isnan(trace[name][outside]).all())
                self.assertEqual(trace["scores"].shape, (n, 3))
                self.assertTrue(np.isnan(trace["scores"][outside]).all())
                result = aggregate_diagnostics(trace, after, blocks, 25)
                self.assertEqual(result["scope"]["analysed_frames"], 120)
                self.assertEqual(result["emission_statistics"]["complete_score_frames"], 120)
                expected_rejected_fraction = 1 / 120 if profile == "reliable" else 0
                self.assertAlmostEqual(result["feature_statistics"]["ball_rejected"]["mean"],
                                       expected_rejected_fraction)


class ReferenceDiagnosticsTests(unittest.TestCase):
    def test_reference_diagnostics_aggregates_splits_merges_boundaries_and_stoppages(self):
        n = 120
        ref = np.full(n, "X", dtype="<U1")
        pred = np.full(n, "X", dtype="<U1")
        # Period 1: frames 10..69 (60 frames)
        # GT: H 10..29 (20f = 0.8s), A 30..49 (20f = 0.8s), H 50..69 (20f = 0.8s)
        ref[10:30] = "H"
        ref[30:50] = "A"
        ref[50:70] = "H"
        # Pred merges the two H spells across the A spell: H 10..69
        pred[10:70] = "H"
        # Period 2: frames 75..114 (40 frames)
        # GT: D 75..84 (10f), A 85..114 (30f = 1.2s)
        ref[75:85] = "D"
        ref[85:115] = "A"
        # Pred splits A 85..114 into two A spells separated by D, with a 2-frame late start:
        # D 75..86, A 87..96 (10f), D 97..100 (4f), A 101..114 (14f)
        pred[75:87] = "D"
        pred[87:97] = "A"
        pred[97:101] = "D"
        pred[101:115] = "A"

        trace = synthetic_trace(n)
        trace["missing"][:] = 0.0
        trace["geometry"][10:30] = 1.0
        trace["geometry"][30:40] = -1.0  # supports A, but pred is H -> suppressed_correct_geometry (10f)
        trace["geometry"][40:50] = 1.0   # opposes A, and pred is H -> selected_wrong_geometry (10f)
        trace["missing"][35:40] = 1.0    # 5 missing frames during wrong team

        blocks = [(1, 10, 69), (2, 75, 114)]
        summary = reference_diagnostics(ref, pred, trace, blocks, 25)
        encoded = json.dumps(summary, allow_nan=False)
        self.assertNotIn("start_frame", encoded)
        self.assertEqual(summary["matching"]["n_gt_spells"], 4)
        self.assertEqual(summary["matching"]["n_pred_spells"], 3)
        self.assertEqual(summary["structural_errors"]["likely_merge_count"], 1)
        self.assertEqual(summary["structural_errors"]["likely_split_count"], 1)
        self.assertEqual(summary["structural_errors"]["wrong_team_overlap_frames"], 20)
        self.assertEqual(summary["structural_errors"]["wrong_team_overlap_reference_spells"], 1)
        self.assertEqual(summary["structural_errors"]["majority_wrong_team_reference_spells"], 1)
        self.assertEqual(summary["duration_bins_seconds"]["0_2s"]["reference_count"], 4)
        self.assertEqual(summary["evidence_associations"]["missing_ball_frames"], 5)
        self.assertEqual(summary["evidence_associations"]["missing_ball_error_frames"], 5)
        self.assertEqual(summary["evidence_associations"]["suppressed_correct_geometry_frames"], 10)
        self.assertEqual(summary["evidence_associations"]["selected_wrong_geometry_frames"], 10)
        self.assertEqual(summary["stoppage"]["missed_dead_frames"], 0)
        self.assertEqual(summary["stoppage"]["false_dead_frames"], 6)
        self.assertEqual(summary["stoppage"]["dead_intervals_iou_05"]["n_matched"], 1)
        self.assertEqual(summary["stoppage"]["restart_transitions_12"]["n_matched"], 1)
        self.assertAlmostEqual(
            summary["stoppage"]["restart_transitions_12"]["latency_frames"]["mean"], 2.0
        )

    def test_reference_diagnostics_rejects_non_x_outside_blocks(self):
        trace = synthetic_trace(6)
        with self.assertRaises(ValueError):
            reference_diagnostics(
                np.array(list("HAAAAX")), np.array(list("XAAAAX")), trace, [(1, 1, 4)], 25
            )


if __name__ == "__main__":
    unittest.main()
