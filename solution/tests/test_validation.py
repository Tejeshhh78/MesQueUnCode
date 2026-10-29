"""Independent-harness contract tests using invented data only; no public GT."""

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from evaluation import possession_metrics as pm
from solution.code import independent_evaluation as validation


FIXTURE = Path(__file__).parent / "fixtures" / "independent_manifest.synthetic.json"


class IndependentValidationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.manifest = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.path = self.root / "manifest.json"

    @property
    def match(self):
        return self.manifest["matches"][0]

    def save(self):
        self.path.write_text(json.dumps(self.manifest), encoding="utf-8")
        return self.path

    def run_manifest(self, **kwargs):
        return validation.evaluate_manifest(self.save(), purpose="development", profiles=("incumbent",), **kwargs)

    def export_inputs(self, evidence="reference", file_format="features_npz"):
        """Invented files exercise declarations; they are not genuine reference data."""
        values = self.match["inputs"]["values"]
        path = self.root / ("observations.npz" if file_format == "features_npz" else "observations.json")
        if file_format == "features_npz":
            np.savez(path, **{key: np.asarray(value) for key, value in values.items()})
        else:
            path.write_text(json.dumps(values), encoding="utf-8")
        self.match["inputs"] = {"format": file_format, "path": path.name}
        labels = self.root / "reference.csv"
        labels.write_text("frame,label\n" + "".join(f"{i},{label}\n" for i, label in enumerate(self.match["labels"]["values"])),
                          encoding="utf-8")
        self.match["labels"] = {"path": labels.name}
        self.match["evidence_type"] = evidence
        provenance = self.match["provenance"]
        provenance.update(label_origin="human_frame_labels" if evidence == "reference" else "automatic_proxy",
                          blocks_origin="period_metadata", input_origin="observed", target_labels_used_in_inputs=False)
        return path, labels

    def test_synthetic_fixture_runs_and_remains_explicitly_non_reference(self):
        report = self.run_manifest()
        self.assertEqual(set(report["strata"]), {"synthetic"})
        self.assertEqual(report["strata"]["synthetic"]["matches"], 1)
        self.assertFalse(report["automatic_promotion"])
        self.assertTrue(report["provenance_is_self_declared"])
        self.assertEqual(report["matches"][0]["metrics"]["incumbent"]["n_graded"], 16)
        self.assertEqual(report["matches"][0]["metrics"]["incumbent"]["n_gt_spells"], 2)
        self.assertTrue(all("confusion" not in metrics for metrics in report["matches"][0]["metrics"].values()))

    def test_profiles_get_independent_input_copies_and_official_paired_metrics(self):
        seen, parsed = [], []
        labels = np.array(self.match["labels"]["values"])
        original_reader = validation._read_labels

        def read_labels(*args):
            parsed.append(True)
            self.assertEqual(len(seen), 2)
            return original_reader(*args)

        def predict(features, events, blocks, pitch, fps, config=None, *, profile="incumbent"):
            self.assertFalse(parsed)
            self.assertEqual(features.mean_speed[0], 3)
            features.mean_speed[0] = -100
            seen.append(profile)
            return (np.full(16, "H") if profile == "incumbent" else labels.copy()), {}

        with mock.patch.object(validation, "predict", new=predict), mock.patch.object(validation, "_read_labels", side_effect=read_labels):
            report = validation.evaluate_manifest(self.save(), purpose="development")
        row = report["matches"][0]
        official = pm.evaluate(pm.label_codes(labels), pm.label_codes(labels))
        self.assertEqual(row["metrics"]["reliable"]["spell_f1_mean"], official["spell_f1_mean"])
        self.assertEqual(row["paired_delta"]["reliable"]["spell_f1_mean"],
                         row["metrics"]["reliable"]["spell_f1_mean"] - row["metrics"]["incumbent"]["spell_f1_mean"])
        json.dumps(report, allow_nan=False)

    def test_adjacent_same_team_periods_remain_two_possessions(self):
        self.match["blocks"] = [{"period": 1, "start": 0, "stop": 8},
                                {"period": 2, "start": 8, "stop": 16}]
        self.match["labels"]["values"] = ["H"] * 16

        def predict(*args, **kwargs):
            return np.full(16, "H"), {}

        with mock.patch.object(validation, "predict", new=predict):
            report = self.run_manifest()
        row = report["matches"][0]
        self.assertEqual(row["metrics"]["incumbent"]["n_gt_spells"], 2)
        self.assertEqual(row["metrics"]["incumbent"]["n_pred_spells"], 2)
        self.assertEqual(row["metrics"]["incumbent"]["transitions"]["25"]["n_gt"], 0)
        self.assertEqual(row["metrics"]["incumbent"]["spell_f1_mean"], 1)
        self.assertEqual(row["evaluation_timebase"]["source_frames"], 16)
        self.assertEqual(row["evaluation_timebase"]["score_grid_frames"], 67)

    def test_period_padding_prevents_spurious_cross_period_transition_match(self):
        reference = np.r_[np.full(58, pm.H), np.full(2, pm.A), np.full(60, pm.H)]
        prediction = np.r_[np.full(62, pm.H), np.full(58, pm.A)]
        # Without period separation, the near-boundary transitions falsely match.
        self.assertEqual(pm.evaluate(reference, prediction)["transitions"][25]["n_matched"], 1)
        gt, predictions, _ = validation._score_grid(reference, {"incumbent": prediction}, [(1, 0, 59), (2, 60, 119)])
        result = pm.evaluate(gt, predictions["incumbent"])
        self.assertEqual(result["transitions"][25]["n_gt"], 1)
        self.assertEqual(result["transitions"][25]["n_pred"], 1)
        self.assertEqual(result["transitions"][25]["n_matched"], 0)
        self.assertIsNone(result["transitions"][25]["mean_latency"])

    def test_metric_padding_preserves_frames_unknown_runs_and_within_period_latencies(self):
        reference = np.r_[np.full(30, pm.H), np.full(30, pm.A), np.full(30, pm.A), np.full(30, pm.H)]
        prediction = np.r_[np.full(32, pm.H), np.full(28, pm.A), np.full(28, pm.A), np.full(32, pm.H)]
        reference[10:14] = pm.X
        gt, predictions, timebase = validation._score_grid(reference, {"incumbent": prediction}, [(1, 0, 59), (2, 60, 119)])
        np.testing.assert_array_equal(gt[:60], reference[:60])
        np.testing.assert_array_equal(gt[111:], reference[60:])
        np.testing.assert_array_equal(predictions["incumbent"][:60], prediction[:60])
        np.testing.assert_array_equal(predictions["incumbent"][111:], prediction[60:])
        self.assertTrue(np.all(gt[60:111] == pm.X))
        self.assertEqual(timebase["inserted_ignored_frames"], 51)
        original = pm.evaluate(reference, prediction)
        expanded = pm.evaluate(gt, predictions["incumbent"])
        self.assertEqual(expanded["n_graded"], original["n_graded"])
        self.assertEqual(expanded["transitions"][25]["n_matched"], 2)
        self.assertEqual(expanded["transitions"][25]["mean_latency"], 0)
        self.assertEqual(expanded["transitions"][25]["mean_abs_latency"], 2)

    def test_npz_and_json_input_contracts_match_the_inline_fixture(self):
        baseline = self.run_manifest()["matches"][0]["metrics"]
        original = copy.deepcopy(self.manifest)
        for file_format in ("features_npz", "features_json"):
            with self.subTest(file_format=file_format):
                self.manifest = copy.deepcopy(original)
                self.export_inputs(file_format=file_format)
                report = self.run_manifest()
                self.assertEqual(report["matches"][0]["metrics"], baseline)
                self.assertEqual(report["matches"][0]["evidence_type"], "reference")

    def test_other_frame_rates_units_shapes_and_period_contracts_are_rejected(self):
        original = copy.deepcopy(self.manifest)
        cases = [lambda m: m.update(fps=50), lambda m: m.update(pitch_m=[0, 68]),
                 lambda m: m.update(units={}), lambda m: m.update(n_frames=True),
                 lambda m: m.update(blocks=[{"period": 1, "start": 1, "stop": 10}, {"period": 2, "start": 9, "stop": 16}]),
                 lambda m: m.update(blocks=[{"period": 1, "start": 0, "stop": 17}]),
                 lambda m: m["inputs"]["values"].update(ball=[[0, 0, 0]]),
                 lambda m: m["inputs"]["values"]["player_count"].__setitem__(0, 1.5),
                 lambda m: m["inputs"]["values"]["distance"][0].__setitem__(0, -1)]
        for change in cases:
            self.manifest = copy.deepcopy(original)
            change(self.match)
            with self.assertRaises(ValueError):
                self.run_manifest()

    def test_competition_identity_and_provenance_are_rejected_before_prediction(self):
        original = copy.deepcopy(self.manifest)
        for change in [lambda m: m.update(match_id="BAR-ATH"),
                       lambda m: m["provenance"].update(competition_match_excluded=False),
                       lambda m: m["provenance"].update(match_identity="The supplied BAR-ATH match"),
                       lambda m: m["provenance"].update(semantics_compatible=False),
                       lambda m: m["provenance"].update(blocks_origin="target_label_boundaries")]:
            self.manifest = copy.deepcopy(original)
            change(self.match)
            with mock.patch.object(validation, "predict") as prediction:
                with self.assertRaises(ValueError):
                    self.run_manifest()
                prediction.assert_not_called()

    def test_competition_gt_path_is_rejected_without_opening_reference_labels(self):
        self.match["labels"] = {"path": str(validation.ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv")}
        with mock.patch.object(validation.pm, "read_label_csv") as reader:
            with self.assertRaises(ValueError):
                self.run_manifest()
            reader.assert_not_called()

    def test_atd_observation_import_preserves_explicit_team_order(self):
        values = self.match["inputs"]["values"]
        metadata, tracking = self.root / "metadata.xml", self.root / "tracking.txt"
        metadata.write_text('<root><Team id="h"><Name>Home</Name></Team>'
                            '<Team id="a"><Name>Away</Name></Team>'
                            '<Player teamId="h"/><Player teamId="a"/></root>', encoding="utf-8")
        rows = []
        for frame, (ball, distance) in enumerate(zip(values["ball"], values["distance"])):
            home = f"{ball[0] + distance[0] / 105},{ball[1]},3"
            away = f"{ball[0] + distance[1] / 105},{ball[1]},3"
            rows.append(f"{frame + 1}:{home};{away}:{ball[0]},{ball[1]},5\n")
        tracking.write_text("".join(rows), encoding="utf-8")
        self.match["inputs"] = {"format": "atd", "metadata": metadata.name, "tracking": tracking.name}
        report = self.run_manifest()
        self.assertEqual(report["matches"][0]["n_frames"], 16)
        self.assertIn("tracking", report["matches"][0]["input_sha256"])
        self.match["teams"] = ["Away", "Home"]
        with self.assertRaisesRegex(ValueError, "team order"):
            self.run_manifest()

    def test_undefined_possession_share_is_not_reported_as_a_real_error(self):
        def predict(*args, **kwargs):
            return np.full(16, "D"), {}

        with mock.patch.object(validation, "predict", new=predict):
            report = self.run_manifest()
        metrics = report["matches"][0]["metrics"]["incumbent"]
        self.assertFalse(metrics["share_defined"])
        self.assertIsNone(metrics["share_err"])
        self.assertNotIn("share_err", report["strata"]["synthetic"]["equal_match_mean"]["incumbent"])

    def test_renamed_known_competition_content_is_rejected_by_hash_guard(self):
        _, labels = self.export_inputs()
        blob = labels.read_bytes().replace(b"\r\n", b"\n")
        digest = hashlib.sha1(f"blob {len(blob)}\0".encode("ascii") + blob).hexdigest()
        with mock.patch.object(validation, "_protected_git_blobs", return_value={digest}), mock.patch.object(validation.pm, "read_label_csv") as reader:
            with self.assertRaises(ValueError):
                self.run_manifest()
            reader.assert_not_called()

    def test_repeated_match_views_and_shared_observations_cannot_cross_splits(self):
        second = copy.deepcopy(self.match)
        second.update(match_id="another-view", split="final_test")
        self.manifest["matches"].append(second)
        with self.assertRaises(ValueError):
            self.run_manifest()
        self.manifest["matches"].pop()
        self.export_inputs()
        second = copy.deepcopy(self.match)
        second.update(match_id="other", group_id="other", split="training")
        self.manifest["matches"].append(second)
        with self.assertRaises(ValueError):
            self.run_manifest()

    def test_development_never_opens_or_requires_unselected_final_test_files(self):
        second = copy.deepcopy(self.match)
        second.update(match_id="unseen-final", group_id="unseen-final", split="final_test", evidence_type="reference")
        second["provenance"].update(label_origin="human_frame_labels", input_origin="observed",
                                     blocks_origin="period_metadata", target_labels_used_in_inputs=False)
        second["inputs"] = {"format": "features_npz", "path": "not-converted-final-inputs.npz"}
        second["labels"] = {"path": "not-opened-final-labels.csv"}
        self.manifest["matches"].append(second)
        report = self.run_manifest()
        self.assertEqual(len(report["matches"]), 1)
        self.assertEqual(report["matches"][0]["match_id"], "invented-match-one")

    def test_reference_evidence_cannot_use_target_labels_to_construct_inputs(self):
        self.export_inputs()
        self.match["provenance"]["target_labels_used_in_inputs"] = True
        with self.assertRaises(ValueError):
            self.run_manifest()

    def test_proxy_and_synthetic_scores_are_not_pooled(self):
        synthetic = copy.deepcopy(self.match)
        synthetic.update(match_id="another-invented-match", group_id="another-invented-group")
        self.export_inputs(evidence="proxy")
        self.manifest["matches"].append(synthetic)
        report = self.run_manifest()
        self.assertEqual(set(report["strata"]), {"proxy", "synthetic"})
        self.assertEqual(report["strata"]["proxy"]["matches"], 1)
        self.assertEqual(report["strata"]["synthetic"]["matches"], 1)
        self.assertNotIn("overall_mean", report)
        self.assertFalse(report["automatic_promotion"])

    def test_final_test_requires_locked_protocol_and_reference_evidence(self):
        self.match["split"] = "final_test"
        with self.assertRaises(ValueError):
            validation.evaluate_manifest(self.save(), purpose="final_test", profiles=("incumbent",))
        self.manifest["protocol"]["final_test_locked"] = True
        with self.assertRaises(ValueError):
            validation.evaluate_manifest(self.save(), purpose="final_test", profiles=("incumbent",))
        self.export_inputs()
        report = validation.evaluate_manifest(self.save(), purpose="final_test", profiles=("incumbent",))
        self.assertEqual(report["purpose"], "final_test")
        self.assertFalse(report["automatic_promotion"])

    def test_explicit_label_coverage_counts_and_masks_are_checked(self):
        original = copy.deepcopy(self.manifest)
        for change in [lambda m: m["labels"]["values"].pop(),
                       lambda m: m["labels"]["values"].__setitem__(0, ""),
                       lambda m: m["provenance"].update(labelled_frames=15),
                       lambda m: m.update(blocks=[{"period": 1, "start": 1, "stop": 16}])]:
            self.manifest = copy.deepcopy(original)
            change(self.match)
            with self.assertRaises(ValueError):
                self.run_manifest()

    def test_npz_object_arrays_are_never_unpickled(self):
        path, _ = self.export_inputs()
        values = dict(np.load(path, allow_pickle=False))
        values["ball"] = np.array([{"unsafe": "object"}], dtype=object)
        np.savez(path, **values)
        with self.assertRaises(ValueError):
            self.run_manifest()

    def test_string_complex_and_boolean_features_are_rejected(self):
        original = copy.deepcopy(self.manifest)
        for dtype in (str, complex, bool):
            self.manifest = copy.deepcopy(original)
            path, _ = self.export_inputs()
            with np.load(path, allow_pickle=False) as archive:
                values = {key: archive[key] for key in archive.files}
            values["ball"] = values["ball"].astype(dtype)
            np.savez(path, **values)
            with self.assertRaisesRegex(ValueError, "numeric"):
                self.run_manifest()
        self.manifest = copy.deepcopy(original)
        self.match["inputs"]["values"]["mean_speed"][0] = "3.0"
        with self.assertRaisesRegex(ValueError, "numbers"):
            self.run_manifest()

    def test_input_changes_during_inference_are_rejected(self):
        _, labels = self.export_inputs()

        def predict(features, *args, **kwargs):
            labels.write_text("frame,label\n" + "".join(f"{i},H\n" for i in range(16)), encoding="utf-8")
            return np.full(16, "H"), {}

        with mock.patch.object(validation, "predict", new=predict):
            with self.assertRaisesRegex(ValueError, "changed during evaluation"):
                self.run_manifest()

    def test_cli_refuses_existing_report_before_any_evaluation(self):
        self.save()
        target = self.root / "synthetic-contract-example-development.json"
        target.write_text("preserve previous results", encoding="utf-8")
        with mock.patch.object(validation, "evaluate_manifest") as evaluate, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                validation.main([str(self.path), "--development", "--output-dir", str(self.root)])
            evaluate.assert_not_called()
        self.assertEqual(target.read_text(encoding="utf-8"), "preserve previous results")

    def test_cli_requires_an_explicit_purpose(self):
        self.save()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                validation.main([str(self.path)])

    def test_cli_writes_aggregate_only_json(self):
        with contextlib.redirect_stdout(io.StringIO()):
            status = validation.main([str(self.save()), "--development", "--profiles", "incumbent",
                                      "--bootstrap", "0", "--output-dir", str(self.root / "reports")])
        self.assertEqual(status, 0)
        report = json.loads((self.root / "reports/synthetic-contract-example-development.json").read_text(encoding="utf-8"))
        row = report["matches"][0]
        self.assertNotIn("labels", row)
        self.assertNotIn("features", row)
        self.assertEqual(len(row["prediction_labels_ascii_sha256"]["incumbent"]), 64)
        for transition in row["metrics"]["incumbent"]["transitions"].values():
            self.assertNotIn("pairs", transition)
            self.assertNotIn("missed", transition)
            self.assertNotIn("spurious", transition)


if __name__ == "__main__":
    unittest.main()
