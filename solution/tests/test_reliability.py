"""Independent synthetic contracts for an opt-in observation reliability experiment."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from solution.code.possession.__main__ import run
from solution.code.possession.data import Features
from solution.code.possession.model import predict, profile_config, reliable_ball_mask


def controlled_sequence(n=300):
    ball = np.column_stack((0.15 + np.arange(n) * 5 / (25 * 105),
                            np.full(n, 0.5), np.full(n, 5.0)))
    return Features(ball, np.tile([0.5, 8.0], (n, 1)), np.full((n, 2), 3.0),
                    np.full(n, 3.0), np.full(n, 0.6), np.full(n, 22), ("Home", "Away"))


class ReliabilityTests(unittest.TestCase):
    def test_isolated_implausible_position_is_removed_without_removing_neighbours(self):
        ball = np.column_stack((np.linspace(.2, .21, 11), np.full(11, .5)))
        ball[5] = [.8, .8]
        mask = reliable_ball_mask(ball, np.ones(11, bool), (105., 68.), 25)
        self.assertEqual(np.flatnonzero(~mask).tolist(), [5])

    def test_plausible_fast_motion_is_kept_at_both_frame_rates(self):
        for fps in (25, 50):
            ball = np.column_stack((.1 + np.arange(40) * 20 / (fps * 105), np.full(40, .5)))
            self.assertTrue(reliable_ball_mask(ball, np.ones(40, bool), (105., 68.), fps).all())

    def test_missing_neighbours_are_not_evidence_of_an_outlier(self):
        ball = np.array([[np.nan, np.nan], [.9, .9], [.2, .5]])
        observed = np.isfinite(ball).all(axis=1)
        np.testing.assert_array_equal(reliable_ball_mask(ball, observed, (105., 68.), 25), observed)

    def test_bad_contracts_rejected(self):
        for fps in (0, -1, np.inf):
            with self.assertRaises(ValueError):
                reliable_ball_mask(np.zeros((3, 2)), np.ones(3, bool), (105., 68.), fps)
        with self.assertRaises(ValueError):
            reliable_ball_mask(np.zeros((3, 3)), np.ones(3, bool), (105., 68.), 25)
        with self.assertRaises(ValueError):
            predict(controlled_sequence(), [], [(1, 0, 299)], (105., 68.), 25, profile="unknown")

    def test_true_short_control_and_brief_deflection_are_distinguished(self):
        features = controlled_sequence()
        features.distance[140:185] = [8., .5]
        features.distance[60:63] = [8., .5]
        for profile in ("incumbent", "reliable"):
            labels, _ = predict(features, [], [(1, 0, 299)], (105., 68.), 25, profile=profile)
            self.assertTrue(np.all(labels[140:185] == "A"))
            self.assertTrue(np.all(labels[60:63] == "H"))

    def test_implausible_flicker_cannot_create_a_stoppage_in_controlled_live_play(self):
        features = controlled_sequence(600)
        features.ball[:, :2] = [.25, .5]
        features.distance[:] = [1., 8.]
        features.ball[150:450:2, 0] = .75
        features.distance[150:450:2] = [20., .1]
        labels, _ = predict(features, [], [(1, 0, 599)], (105., 68.), 25, profile="reliable")
        self.assertTrue(np.all(labels == "H"))

    def test_team_swap_and_pitch_reflection_preserve_non_tied_decisions(self):
        features = controlled_sequence()
        features.distance[140:185] = [8., .5]
        reflected = features.ball.copy()
        reflected[:, :2] = 1 - reflected[:, :2]
        for profile in ("incumbent", "reliable"):
            expected, _ = predict(features, [], [(1, 0, 299)], (105., 68.), 25, profile=profile)
            mirror, _ = predict(replace(features, ball=reflected), [], [(1, 0, 299)],
                                (105., 68.), 25, profile=profile)
            np.testing.assert_array_equal(mirror, expected)
            swapped, _ = predict(replace(features, distance=features.distance[:, ::-1]), [],
                                 [(1, 0, 299)], (105., 68.), 25, profile=profile)
            np.testing.assert_array_equal(swapped, np.where(expected == "H", "A", np.where(expected == "A", "H", expected)))

    def test_candidate_cannot_overwrite_incumbent_output(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            target = directory / "Test.csv"
            target.write_text("frame,label\n0,H\n", encoding="utf-8")
            (directory / "Test.manifest.json").write_text(json.dumps({"profile": "incumbent"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different model profile"):
                run(None, directory, "Test", profile="reliable")
            self.assertEqual(target.read_text(encoding="utf-8"), "frame,label\n0,H\n")

    def test_changed_configuration_cannot_overwrite_a_previous_experiment(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            target = directory / "Test.csv"
            target.write_text("frame,label\n0,H\n", encoding="utf-8")
            manifest = {"profile": "responsive", "config": profile_config("responsive").to_dict()}
            (directory / "Test.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "configuration"):
                run(None, directory, "Test", profile="responsive", team_switch_cost=4)
            self.assertEqual(target.read_text(encoding="utf-8"), "frame,label\n0,H\n")

    def test_cli_cost_override_is_explicit_and_finite(self):
        with tempfile.TemporaryDirectory() as folder:
            for profile, cost in (("incumbent", 4), ("reliable", 4),
                                  ("responsive", -1), ("responsive", 0),
                                  ("responsive", float("nan")), ("responsive", float("inf"))):
                with self.assertRaisesRegex(ValueError, "positive finite"):
                    run(None, folder, "Test", profile=profile, team_switch_cost=cost)


if __name__ == "__main__":
    unittest.main()
