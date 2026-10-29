"""Independent synthetic contracts for the opt-in team-switch-cost experiment.

These are feature-level scenarios and invented decoder scores, not observations
or labels from any competition or external match. They test model behaviour,
not real-match possession accuracy.
"""

from dataclasses import replace
import unittest
from unittest import mock

import numpy as np

from solution.code.possession.data import Features
from solution.code.possession import model


PITCH = (105.0, 68.0)
FPS = 25
COSTS = (4.0, 8.0, 12.0)


def control_features(n=300):
    """Create a moving ball with clear home-team control evidence."""
    ball = np.column_stack((0.15 + np.arange(n) * 5 / (FPS * PITCH[0]),
                            np.full(n, 0.5), np.full(n, 5.0)))
    return Features(ball, np.tile([0.5, 8.0], (n, 1)), np.full((n, 2), 3.0),
                    np.full(n, 3.0), np.full(n, 0.6), np.full(n, 22), ("Home", "Away"))


def candidate_config(cost):
    return replace(model.profile_config("responsive"), team_switch_cost=cost)


class ResponsiveControlTests(unittest.TestCase):
    def test_responsive_default_changes_only_the_declared_team_switch_cost(self):
        incumbent = model.profile_config("incumbent").to_dict()
        responsive = model.profile_config("responsive").to_dict()
        self.assertEqual(incumbent.pop("team_switch_cost"), 20.0)
        self.assertEqual(responsive.pop("team_switch_cost"), 12.0)
        self.assertEqual(responsive, incumbent)

    def test_same_config_produces_exactly_identical_incumbent_and_responsive_evidence(self):
        features = control_features()
        features.ball[40:45] = np.nan
        features.distance[40:45] = np.nan
        features.mean_speed[125:175] = 0.3
        events = [
            {"start": 10, "stop": 100, "code": "buildup", "tags": {"team": "Home"}},
            {"start": 180, "stop": 210, "code": "setpieces", "tags": {"team": "Away"}},
        ]
        blocks = [(1, 10, 139), (2, 170, 289)]
        for cost in COSTS:
            with self.subTest(cost=cost):
                config = candidate_config(cost)
                old_trace, new_trace = {}, {}
                old, _ = model.emissions(features, events, blocks, PITCH, FPS, config,
                                         profile="incumbent", trace=old_trace)
                new, _ = model.emissions(features, events, blocks, PITCH, FPS, config,
                                         profile="responsive", trace=new_trace)
                np.testing.assert_array_equal(old, new)
                self.assertEqual(set(old_trace), set(new_trace))
                for name in old_trace:
                    np.testing.assert_array_equal(old_trace[name], new_trace[name])

    def test_all_predeclared_costs_preserve_a_short_strong_control_interval(self):
        features = control_features()
        features.distance[140:152] = [8.0, 0.5]
        blocks = [(1, 0, 299)]
        incumbent, _ = model.predict(features, [], blocks, PITCH, FPS)
        self.assertTrue(np.all(incumbent == "H"))
        for cost in COSTS:
            with self.subTest(cost=cost):
                labels, _ = model.predict(features, [], blocks, PITCH, FPS,
                                           candidate_config(cost), profile="responsive")
                np.testing.assert_array_equal(labels[:140], "H")
                np.testing.assert_array_equal(labels[140:152], "A")
                np.testing.assert_array_equal(labels[152:], "H")

    def test_three_frame_glances_and_weak_longer_evidence_do_not_create_phases(self):
        features = control_features()
        features.distance[60:63] = [8.0, 0.5]
        features.distance[140:152] = [8.0, 6.0]
        for cost in COSTS:
            with self.subTest(cost=cost):
                labels, _ = model.predict(features, [], [(1, 0, 299)], PITCH, FPS,
                                           candidate_config(cost), profile="responsive")
                self.assertTrue(np.all(labels == "H"))

    def test_explicit_configuration_overrides_the_profile_default(self):
        features = control_features()
        features.distance[140:152] = [8.0, 0.5]
        blocks = [(1, 0, 299)]
        default, _ = model.predict(features, [], blocks, PITCH, FPS, profile="responsive")
        high_cost, _ = model.predict(features, [], blocks, PITCH, FPS,
                                     model.Config(), profile="responsive")
        low_cost_incumbent, _ = model.predict(features, [], blocks, PITCH, FPS,
                                              candidate_config(4.0), profile="incumbent")
        np.testing.assert_array_equal(default[140:152], "A")
        np.testing.assert_array_equal(high_cost, "H")
        np.testing.assert_array_equal(low_cost_incumbent[140:152], "A")

    def test_stoppage_decisions_keep_the_original_two_switch_cost(self):
        # Isolate H-D-H decisions from team-control evidence. Ten frames with
        # total advantages 47 and 49 bracket the unchanged 24+24 stopping cost.
        # Patching only evidence leaves the real predict/decoder/config wiring
        # under test; no transition-cost construction is duplicated here.
        features = control_features(80)
        blocks = [(1, 0, 79)]
        for advantage in (4.7, 4.9):
            scores = np.tile([0.0, -100.0, -100.0], (80, 1))
            scores[30:40, 2] = advantage
            expected = np.full(80, "H")
            if advantage == 4.9:
                expected[30:40] = "D"
            for cost in COSTS:
                with self.subTest(advantage=advantage, cost=cost), \
                        mock.patch.object(model, "emissions", return_value=(scores, {})):
                    labels, _ = model.predict(features, [], blocks, PITCH, FPS,
                                               candidate_config(cost), profile="responsive")
                    np.testing.assert_array_equal(labels, expected)

    def test_half_boundaries_reset_control_and_keep_ungraded_frames_as_x(self):
        features = control_features()
        features.distance[170:290] = [8.0, 0.5]
        blocks = [(1, 10, 129), (2, 170, 289)]
        for cost in COSTS:
            with self.subTest(cost=cost):
                labels, _ = model.predict(features, [], blocks, PITCH, FPS,
                                           candidate_config(cost), profile="responsive")
                np.testing.assert_array_equal(labels[10:130], "H")
                np.testing.assert_array_equal(labels[170:290], "A")
                outside = np.ones(300, dtype=bool)
                outside[10:130] = False
                outside[170:290] = False
                np.testing.assert_array_equal(labels[outside], "X")

    def test_team_swap_and_pitch_reflection_preserve_non_tied_control_decisions(self):
        features = control_features()
        features.distance[140:152] = [8.0, 0.5]
        reflected = features.ball.copy()
        reflected[:, :2] = 1.0 - reflected[:, :2]
        blocks = [(1, 0, 299)]
        for cost in COSTS:
            with self.subTest(cost=cost):
                config = candidate_config(cost)
                expected, _ = model.predict(features, [], blocks, PITCH, FPS, config, profile="responsive")
                mirror, _ = model.predict(replace(features, ball=reflected), [], blocks, PITCH, FPS,
                                           config, profile="responsive")
                np.testing.assert_array_equal(mirror, expected)
                swapped, _ = model.predict(replace(features, distance=features.distance[:, ::-1]), [],
                                            blocks, PITCH, FPS, config, profile="responsive")
                np.testing.assert_array_equal(swapped, np.where(expected == "H", "A",
                                                                 np.where(expected == "A", "H", expected)))


if __name__ == "__main__":
    unittest.main()
