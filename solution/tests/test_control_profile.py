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


class ControlledEvidenceProfileTests(unittest.TestCase):
    def test_controlled_default_uses_cost_12_and_preserves_short_strong_control_vs_glance(self):
        self.assertEqual(model.profile_config("controlled").team_switch_cost, 12.0)
        features = control_features()
        features.distance[60:63] = [8.0, 0.5]
        features.distance[140:152] = [8.0, 0.5]
        features.distance[200:220] = [8.0, 6.0]
        labels, _ = model.predict(features, [], [(1, 0, 299)], PITCH, FPS, profile="controlled")
        np.testing.assert_array_equal(labels[:140], "H")
        np.testing.assert_array_equal(labels[140:152], "A")
        np.testing.assert_array_equal(labels[152:], "H")

    def test_contested_reception_and_pass_flight_past_opponent_recover_short_spell(self):
        features = control_features()
        # Away receives under pressure (100..113) and then passes past a nearby Home defender (114..139)
        features.distance[100:114] = [1.8, 0.5]
        features.distance[114:132] = [2.6, 5.5]
        features.distance[132:140] = [5.0, 0.5]
        for t in range(114, 132):
            features.ball[t, 0] = features.ball[t - 1, 0] + 13.0 / (FPS * PITCH[0])
            features.ball[t, 2] = 13.0
        for t in range(132, 300):
            features.ball[t, 0] = features.ball[t - 1, 0] + 5.0 / (FPS * PITCH[0])
            features.ball[t, 2] = 5.0
        ctrl_labels, _ = model.predict(features, [], [(1, 0, 299)], PITCH, FPS, profile="controlled")
        np.testing.assert_array_equal(ctrl_labels[:100], "H")
        np.testing.assert_array_equal(ctrl_labels[100:140], "A")
        np.testing.assert_array_equal(ctrl_labels[140:], "H")

    def test_fast_pass_flying_past_opponent_does_not_flip_possession(self):
        features = control_features()
        # Home passes at 15 m/s; ball flies past an Away defender at 2.2 m for 12 frames (120..131)
        features.distance[120:132] = [5.5, 2.2]
        for t in range(120, 132):
            features.ball[t, 0] = features.ball[t - 1, 0] + 15.0 / (FPS * PITCH[0])
            features.ball[t, 2] = 15.0
        for t in range(132, 300):
            features.ball[t, 0] = features.ball[t - 1, 0] + 5.0 / (FPS * PITCH[0])
            features.ball[t, 2] = 5.0
        labels, _ = model.predict(features, [], [(1, 0, 299)], PITCH, FPS, profile="controlled")
        self.assertTrue(np.all(labels == "H"))

    def test_stationary_live_ball_vs_low_activity_stoppage(self):
        live = control_features(400)
        live.ball[130:220, :2] = live.ball[130, :2]
        live.ball[130:220, 2] = 0.0
        live.mean_speed[:] = 3.5  # high collective speed keeps play alive
        live_labels, _ = model.predict(live, [], [(1, 0, 399)], PITCH, FPS, profile="controlled")
        self.assertTrue(np.all(live_labels == "H"))

        dead = control_features(400)
        dead.ball[130:260, :2] = dead.ball[130, :2]
        dead.ball[130:260, 2] = 0.0
        dead.mean_speed[130:260] = 0.3  # low collective speed + stationary ball -> stoppage
        dead_labels, _ = model.predict(dead, [], [(1, 0, 399)], PITCH, FPS, profile="controlled")
        self.assertTrue(np.all(dead_labels[160:230] == "D"))
        self.assertTrue(np.all(dead_labels[280:350] == "H"))

    def test_missing_observations_preserve_or_switch_ownership_appropriately(self):
        same = control_features(300)
        same.ball[100:160] = np.nan
        same.distance[100:160] = np.nan
        same_labels, _ = model.predict(same, [], [(1, 0, 299)], PITCH, FPS, profile="controlled")
        self.assertTrue(np.all(same_labels == "H"))

        turn = control_features(300)
        turn.ball[100:130] = np.nan
        turn.distance[100:130] = np.nan
        turn.distance[130:] = [8.0, 0.5]
        turn_labels, _ = model.predict(turn, [], [(1, 0, 299)], PITCH, FPS, profile="controlled")
        self.assertTrue(np.all(turn_labels[:100] == "H"))
        self.assertTrue(np.all(turn_labels[135:] == "A"))

    def test_half_resets_team_swap_pitch_reflection_and_deterministic_ties(self):
        features = control_features()
        features.distance[170:290] = [8.0, 0.5]
        blocks = [(1, 10, 129), (2, 170, 289)]
        labels, _ = model.predict(features, [], blocks, PITCH, FPS, profile="controlled")
        np.testing.assert_array_equal(labels[10:130], "H")
        np.testing.assert_array_equal(labels[170:290], "A")
        np.testing.assert_array_equal(labels[:10], "X")
        np.testing.assert_array_equal(labels[130:170], "X")
        np.testing.assert_array_equal(labels[290:], "X")

        reflected = features.ball.copy()
        reflected[:, :2] = 1.0 - reflected[:, :2]
        mirror, _ = model.predict(replace(features, ball=reflected), [], blocks, PITCH, FPS, profile="controlled")
        np.testing.assert_array_equal(mirror, labels)

        swapped_feat = replace(
            features,
            distance=features.distance[:, ::-1],
            player_speed=features.player_speed[:, ::-1],
        )
        swapped, _ = model.predict(swapped_feat, [], blocks, PITCH, FPS, profile="controlled")
        np.testing.assert_array_equal(
            swapped,
            np.where(labels == "H", "A", np.where(labels == "A", "H", labels)),
        )

        tied = control_features(60)
        tied.distance[:] = [3.0, 3.0]
        trace = {}
        tied_labels, _ = model.predict(tied, [], [(1, 0, 59)], PITCH, FPS, profile="controlled", trace=trace)
        np.testing.assert_array_equal(trace["geometry"], np.zeros(60))
        np.testing.assert_array_equal(tied_labels, np.full(60, "H"))


if __name__ == "__main__":
    unittest.main()
