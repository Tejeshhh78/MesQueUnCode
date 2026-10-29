"""Synthetic decision/locking tests; no real match predictions or evaluation."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from solution.code import control_experiment as experiment


def protocol():
    return {"id": "control-switch-cost-002", "declared_before_results": True,
            "development": ["invented-dev-one", "invented-dev-two"],
            "final_test": ["invented-final-one", "invented-final-two"], **copy.deepcopy(experiment.POLICY)}


def fake_report(policy, cost=8.0, purpose="development"):
    """Invented metric records exercise selection, not football quality."""
    rows = []
    for index, match_id in enumerate(policy[purpose]):
        base = 0.4 + index * 0.05
        def metrics(primary):
            return {"spell_f1_mean": primary, "macro_f1": 0.8, "per_class": {"D": {"recall": 0.9}},
                    "spell_f1": {0.5: {"recall": 0.7}}, "transitions": {"12": {"f1": 0.6}}}
        rows.append({"match_id": match_id, "evidence_type": "reference",
                     "metrics": {"incumbent": metrics(base), "responsive": metrics(base + (0.03 if cost == 4 else 0.05))},
                     "predict_seconds": {"incumbent": 1., "responsive": 1.1},
                     "input_sha256": {"invented": "a" * 64},
                     "prediction_labels_ascii_sha256": {"incumbent": "b" * 64, "responsive": "c" * 64}})
    return {"purpose": purpose, "matches": rows}


class SelectionRuleTests(unittest.TestCase):
    def test_selects_best_mean_and_prefers_higher_cost_on_exact_tie(self):
        policy = protocol()
        assessments = {str(cost): experiment.assess_candidate(fake_report(policy, cost), policy)
                       for cost in policy["candidate_costs"]}
        self.assertEqual(experiment.select_result(assessments), 12)

    def test_per_match_regression_cannot_hide_behind_a_higher_mean(self):
        policy, report = protocol(), fake_report(protocol())
        report["matches"][0]["metrics"]["responsive"]["spell_f1_mean"] = 0.39
        report["matches"][1]["metrics"]["responsive"]["spell_f1_mean"] = 0.9
        result = experiment.assess_candidate(report, policy)
        self.assertTrue(result["mean_gain_gate"])
        self.assertFalse(result["eligible"])
        self.assertFalse(result["per_match"][0]["gates"]["nonnegative_primary"])

    def test_each_secondary_and_runtime_gate_is_enforced(self):
        for key, mutate in [
            ("macro_limit", lambda r: r["metrics"]["responsive"].update(macro_f1=0.794)),
            ("dead_recall_limit", lambda r: r["metrics"]["responsive"]["per_class"]["D"].update(recall=0.879)),
            ("iou_recall_limit", lambda r: r["metrics"]["responsive"]["spell_f1"][0.5].update(recall=0.679)),
            ("transition_limit", lambda r: r["metrics"]["responsive"]["transitions"]["12"].update(f1=0.579)),
            ("runtime_limit", lambda r: r["predict_seconds"].update(responsive=1.501)),
        ]:
            with self.subTest(key=key):
                report = fake_report(protocol())
                mutate(report["matches"][0])
                result = experiment.assess_candidate(report, protocol())
                self.assertFalse(result["eligible"])
                self.assertFalse(result["per_match"][0]["gates"][key])

    def test_minimum_mean_gain_and_exact_limit_boundaries(self):
        policy, report = protocol(), fake_report(protocol())
        for row in report["matches"]:
            old = row["metrics"]["incumbent"]
            row["metrics"]["responsive"]["spell_f1_mean"] = old["spell_f1_mean"] + 0.009
        self.assertFalse(experiment.assess_candidate(report, policy)["eligible"])
        for row in report["matches"]:
            old, new = row["metrics"]["incumbent"], row["metrics"]["responsive"]
            new["spell_f1_mean"] = old["spell_f1_mean"] + 0.01
            new["macro_f1"] = old["macro_f1"] - 0.005
            new["per_class"]["D"]["recall"] = 0.88
            new["spell_f1"][0.5]["recall"] = 0.68
            new["transitions"]["12"]["f1"] = 0.58
            row["predict_seconds"]["responsive"] = 1.5
        self.assertTrue(experiment.assess_candidate(report, policy)["eligible"])

    def test_proxy_missing_match_nonfinite_metrics_and_zero_time_are_rejected(self):
        for mutate in [lambda r: r["matches"].pop(),
                       lambda r: r["matches"][0].update(evidence_type="proxy"),
                       lambda r: r["matches"][0]["metrics"]["responsive"].update(macro_f1=float("nan")),
                       lambda r: r["matches"][0]["predict_seconds"].update(incumbent=0)]:
            report = fake_report(protocol())
            mutate(report)
            with self.assertRaises(ValueError):
                experiment.assess_candidate(report, protocol())


class LockedExperimentTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.policy = protocol()
        self.protocol = self.root / "protocol.json"
        self.protocol.write_text(json.dumps(self.policy), encoding="utf-8")
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps({"matches": [
            {"match_id": identity, "split": split, "evidence_type": "reference"}
            for split in ("development", "final_test") for identity in self.policy[split]]}), encoding="utf-8")
        self.output = self.root / "development"
        self.sources = {"synthetic-decision-test-only.py": "d" * 64}
        patcher = mock.patch.object(experiment, "_source_snapshot", side_effect=lambda: copy.deepcopy(self.sources))
        patcher.start()
        self.addCleanup(patcher.stop)

    def evaluate(self, path, *, purpose, profiles, configs):
        self.assertEqual(tuple(profiles), ("incumbent", "responsive"))
        self.assertEqual(configs["incumbent"].team_switch_cost, 20)
        self.assertEqual(configs["responsive"].stoppage_switch_cost, 24)
        return fake_report(self.policy, configs["responsive"].team_switch_cost, purpose)

    def select(self, evaluate=None):
        with mock.patch.object(experiment.validation, "evaluate_manifest", side_effect=evaluate or self.evaluate) as evaluator:
            result = experiment.run_development(self.protocol, self.manifest, self.output)
        return result, evaluator

    def test_fixed_grid_locks_sources_manifest_protocol_and_selected_configuration(self):
        result, evaluator = self.select()
        self.assertEqual(evaluator.call_count, 3)
        self.assertEqual(result["selected_switch_cost"], 12)
        self.assertEqual(result["configurations"]["responsive"]["team_switch_cost"], 12)
        self.assertFalse(result["automatic_promotion"])
        self.assertEqual(result["source_sha256"], self.sources)
        self.assertEqual(result["manifest_sha256"], experiment._file_hash(self.manifest))
        self.assertEqual(result["protocol_sha256"], experiment._file_hash(self.protocol))
        self.assertTrue((self.output / "candidate-4.json").exists())
        with self.assertRaises(FileExistsError):
            experiment.run_development(self.protocol, self.manifest, self.output)

    def test_same_protocol_cannot_repeat_grid_in_another_output_directory(self):
        self.select()
        with mock.patch.object(experiment.validation, "evaluate_manifest") as evaluator:
            with self.assertRaises(FileExistsError):
                experiment.run_development(self.protocol, self.manifest, self.root / "other-development")
            evaluator.assert_not_called()

    def test_no_qualified_candidate_keeps_incumbent_and_never_opens_final_evaluation(self):
        def no_gain(path, **kwargs):
            report = self.evaluate(path, **kwargs)
            for row in report["matches"]:
                row["metrics"]["responsive"]["spell_f1_mean"] = row["metrics"]["incumbent"]["spell_f1_mean"]
            return report

        result, _ = self.select(no_gain)
        self.assertEqual(result["decision"], "retain_incumbent")
        with mock.patch.object(experiment.validation, "evaluate_manifest") as evaluator:
            with self.assertRaisesRegex(ValueError, "No candidate qualified"):
                experiment.run_final_test(self.output / "selection.json", self.root / "final")
            evaluator.assert_not_called()

    def test_source_manifest_or_protocol_changes_block_final_before_evaluation(self):
        self.select()
        self.sources["synthetic-decision-test-only.py"] = "e" * 64
        with mock.patch.object(experiment.validation, "evaluate_manifest") as evaluator:
            with self.assertRaisesRegex(ValueError, "changed"):
                experiment.run_final_test(self.output / "selection.json", self.root / "final")
            evaluator.assert_not_called()
        self.sources["synthetic-decision-test-only.py"] = "d" * 64
        self.manifest.write_text(self.manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with mock.patch.object(experiment.validation, "evaluate_manifest") as evaluator:
            with self.assertRaisesRegex(ValueError, "changed"):
                experiment.run_final_test(self.output / "selection.json", self.root / "final")
            evaluator.assert_not_called()

    def test_changed_selection_config_is_detected(self):
        self.select()
        path = self.output / "selection.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["configurations"]["responsive"]["team_switch_cost"] = 4
        path.write_text(json.dumps(record), encoding="utf-8")
        with mock.patch.object(experiment.validation, "evaluate_manifest") as evaluator:
            with self.assertRaisesRegex(ValueError, "selection record"):
                experiment.run_final_test(path, self.root / "final")
            evaluator.assert_not_called()

    def test_final_runs_once_with_exact_selected_cost_and_never_promotes_automatically(self):
        self.select()
        def final(path, **kwargs):
            self.assertEqual(kwargs["configs"]["responsive"].team_switch_cost, 12)
            self.assertEqual(kwargs["purpose"], "final_test")
            return self.evaluate(path, **kwargs)

        with mock.patch.object(experiment.validation, "evaluate_manifest", side_effect=final) as evaluator:
            result = experiment.run_final_test(self.output / "selection.json", self.root / "final")
            self.assertEqual(evaluator.call_count, 1)
        self.assertFalse(result["automatic_promotion"])
        self.assertEqual(result["review_recommendation"], "candidate_supported_for_parent_review")
        with mock.patch.object(experiment.validation, "evaluate_manifest") as evaluator:
            with self.assertRaises(FileExistsError):
                experiment.run_final_test(self.output / "selection.json", self.root / "different-final-output")
            evaluator.assert_not_called()

    def test_copied_selection_directory_cannot_reopen_same_held_out_dataset(self):
        self.select()
        with mock.patch.object(experiment.validation, "evaluate_manifest", side_effect=self.evaluate):
            experiment.run_final_test(self.output / "selection.json", self.root / "final")
        copied = self.root / "copied-selection"
        copied.mkdir()
        (copied / "selection.json").write_bytes((self.output / "selection.json").read_bytes())
        with mock.patch.object(experiment.validation, "evaluate_manifest") as evaluator:
            with self.assertRaises(FileExistsError):
                experiment.run_final_test(copied / "selection.json", self.root / "another-final")
            evaluator.assert_not_called()

    def test_final_failure_keeps_incumbent_without_retuning(self):
        self.select()
        def regression(path, **kwargs):
            report = self.evaluate(path, **kwargs)
            report["matches"][0]["metrics"]["responsive"]["macro_f1"] = 0.7
            return report

        with mock.patch.object(experiment.validation, "evaluate_manifest", side_effect=regression):
            result = experiment.run_final_test(self.output / "selection.json", self.root / "final")
        self.assertFalse(result["assessment"]["eligible"])
        self.assertEqual(result["review_recommendation"], "retain_incumbent")

    def test_changed_incumbent_across_grid_is_rejected(self):
        calls = []
        def changed(path, **kwargs):
            report = self.evaluate(path, **kwargs)
            calls.append(True)
            if len(calls) > 1:
                report["matches"][0]["metrics"]["incumbent"]["macro_f1"] = 0.7
            return report

        with self.assertRaisesRegex(ValueError, "incumbent predictions differ"):
            self.select(changed)


if __name__ == "__main__":
    unittest.main()
