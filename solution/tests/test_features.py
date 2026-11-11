"""Per-frame evidence: missing ball, team assignment, dead-ball signals, Smart Tagging."""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from solution import config, features, loaders  # noqa: E402
from solution.loaders import AWAY, HOME, TagEvent  # noqa: E402
from solution.tests import helpers  # noqa: E402

PARAMS = config.DEFAULT_PARAMS


class MissingBallTest(unittest.TestCase):
    def test_short_gaps_are_interpolated_long_ones_are_not(self):
        values = np.array([0.0, np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, np.nan, 8.0])
        out = features.interpolate_short_gaps(values, max_gap=2)
        np.testing.assert_allclose(out[:4], [0, 1, 2, 3])
        self.assertTrue(np.isnan(out[4:8]).all())

    def test_gaps_at_the_edges_are_left_alone(self):
        out = features.interpolate_short_gaps(np.array([np.nan, 1.0, np.nan]), max_gap=5)
        self.assertTrue(np.isnan(out[0]) and np.isnan(out[2]))

    def test_no_ball_means_no_team_evidence(self):
        nan = np.full(3, np.nan)
        evidence = features.control_evidence(nan, nan, nan, PARAMS)
        np.testing.assert_array_equal(evidence, 0.0)

    def test_evidence_never_nan_when_ball_is_missing(self):
        evidence = features.build_evidence(helpers.make_match(), PARAMS, helpers.PITCH)
        self.assertFalse(np.isnan(evidence.team).any())
        self.assertFalse(np.isnan(evidence.dead).any())

    def test_speed_needs_both_ends(self):
        x = np.array([0.0, 1.0, np.nan, 3.0, 4.0, 5.0, 6.0])
        speed = features.ball_speed(x, np.zeros_like(x), halfwidth=1, fps=25)
        self.assertTrue(np.isnan(speed[1]) and np.isnan(speed[3]))
        self.assertAlmostEqual(speed[4], 25.0)


class TeamAssignmentTest(unittest.TestCase):
    def test_nearest_player_per_team(self):
        tracking = helpers.make_tracking(10)
        helpers.put_ball_at_player(tracking, 2, 0, 10)  # an away player's feet
        home, away = features.nearest_by_team(tracking.players, tracking.team,
                                              tracking.ball[:, 0].astype(float),
                                              tracking.ball[:, 1].astype(float))
        self.assertTrue(np.all(away < 0.5))
        self.assertTrue(np.all(home > 10))
        np.testing.assert_array_equal(features.nearest_player_codes(tracking), AWAY)

    def test_close_slow_ball_favours_the_close_team_fast_ball_less(self):
        slow = features.control_evidence(np.array([0.3]), np.array([8.0]), np.array([2.0]), PARAMS)
        fast = features.control_evidence(np.array([0.3]), np.array([8.0]), np.array([20.0]), PARAMS)
        self.assertGreater(slow[0], 0)
        self.assertGreater(slow[0], fast[0])
        flipped = features.control_evidence(np.array([8.0]), np.array([0.3]), np.array([2.0]),
                                            PARAMS)
        self.assertAlmostEqual(flipped[0], -slow[0])

    def test_tag_team_names_are_matched_case_insensitively(self):
        self.assertEqual(loaders.team_code("barcelona ", "Barcelona", "Atletic Club"), HOME)
        self.assertEqual(loaders.team_code("ATLETIC CLUB", "Barcelona", "Atletic Club"), AWAY)
        self.assertIsNone(loaders.team_code("N/A", "Barcelona", "Atletic Club"))
        self.assertIsNone(loaders.team_code(None, "Barcelona", "Atletic Club"))

    def test_only_possession_codes_feed_the_tag_prior(self):
        tags = [TagEvent("BUILD UP", 0, 5, HOME, None),
                TagEvent("DEFENDING IN ATTACKING THIRD", 0, 5, AWAY, None),
                TagEvent("PROGRESSION", 5, 10, AWAY, None),
                TagEvent("PROGRESSION", 10, 12, None, None)]
        prior = features.tag_team_prior(tags, 12, PARAMS)
        self.assertTrue(np.all(prior[:5] > 0))
        self.assertTrue(np.all(prior[5:10] < 0))
        np.testing.assert_array_equal(prior[10:], 0.0)


class DeadBallTest(unittest.TestCase):
    def test_out_of_pitch_uses_a_margin(self):
        x = np.array([-2.0, -0.2, 50.0, 106.5, 50.0, np.nan])
        y = np.array([30.0, 30.0, 68.3, 30.0, 70.0, 30.0])
        out = features.out_of_pitch(x, y, helpers.PITCH, margin=0.75)
        np.testing.assert_array_equal(out, [True, False, False, True, True, False])

    def test_a_ball_last_seen_out_stays_out_through_the_gap(self):
        out = np.array([False, True, False, False, False, False])
        known = np.array([True, True, False, False, True, True])
        carried = features.carry_out_through_gaps(out, known)
        np.testing.assert_array_equal(carried, [False, True, True, True, False, False])

    def test_ball_out_produces_dead_evidence(self):
        evidence = features.build_evidence(helpers.possession_script(), PARAMS, helpers.PITCH)
        self.assertTrue(np.all(evidence.dead[800:900] > 0))

    def test_set_piece_window_opens_with_dead_prior(self):
        tags = [TagEvent("SET PIECES", 10, 10 + 35 * 25, None, "throw-in"),
                TagEvent("SET PIECES", 1000, 1100, HOME, "penalty")]
        prior = features.set_piece_dead_prior(tags, 1200, PARAMS, fps=25)
        self.assertGreater(prior[10], 0)
        self.assertEqual(prior[9], 0)
        self.assertEqual(prior[10 + int(PARAMS.set_piece_dead_seconds * 25)], 0)
        np.testing.assert_array_equal(prior[1000:], 0.0)

    def test_kickoff_forces_dead_before_it_in_its_half(self):
        tracking = helpers.make_tracking()
        tracking.ball[600:640, 0] = helpers.PITCH[0] / 2
        tracking.ball[600:640, 1] = helpers.PITCH[1] / 2
        tracking.ball[640:700, 0] = helpers.PITCH[0] / 2 + np.arange(60) * 0.4  # 10 m/s
        tracking.ball[640:700, 1] = helpers.PITCH[1] / 2
        tags = [TagEvent("SET PIECES", 580, 700, None, "kick off (start)")]
        evidence = features.build_evidence(helpers.make_match(tracking, tags), PARAMS,
                                           helpers.PITCH)
        self.assertEqual(len(evidence.kickoffs), 1)
        kickoff = evidence.kickoffs[0]
        self.assertTrue(630 <= kickoff <= 645)
        self.assertTrue(evidence.forced_dead[600:kickoff].all())
        self.assertFalse(evidence.forced_dead[kickoff:].any())
        self.assertFalse(evidence.forced_dead[:600].any())


class ShapeModelTest(unittest.TestCase):
    def test_logistic_regression_learns_a_separable_rule(self):
        rng = np.random.RandomState(0)
        x = rng.normal(size=(400, 2))
        y = (x[:, 0] > 0).astype(float)
        weights = features.fit_logistic(x, y, l2=1.0)
        self.assertGreater(weights[1], 1.0)
        self.assertLess(abs(weights[2]), abs(weights[1]))

    def test_no_shape_model_gives_no_shape_evidence(self):
        match = helpers.make_match()
        params = PARAMS.with_overrides(shape_mean=(), shape_std=(), shape_weights=())
        out = features.shape_log_odds(match.tracking, match.halves, params, 25, helpers.PITCH)
        np.testing.assert_array_equal(out, 0.0)

    def test_malformed_shape_model_is_an_error(self):
        match = helpers.make_match()
        params = PARAMS.with_overrides(shape_weights=(0.1, 0.2))
        with self.assertRaises(ValueError):
            features.shape_log_odds(match.tracking, match.halves, params, 25, helpers.PITCH)


class RuleComplianceTest(unittest.TestCase):
    """The organisers' clarification: nothing may be fitted or tuned on the provided match."""

    def test_inference_never_fits_anything(self):
        def refuse(*args, **kwargs):
            raise AssertionError("fit_logistic was called at inference time")
        original = features.fit_logistic
        features.fit_logistic = refuse
        try:
            features.build_evidence(helpers.possession_script(), PARAMS, helpers.PITCH)
        finally:
            features.fit_logistic = original

    def test_calibration_comes_from_external_data(self):
        provenance = config.CALIBRATION["provenance"]
        self.assertFalse(provenance["provided_match_used"])
        self.assertIn("metrica-sports/sample-data", provenance["data"])
        self.assertEqual(len(config.CALIBRATION["shape"]["weights"]), len(features.SHAPE_FEATURES) + 1)
        self.assertEqual(tuple(config.CALIBRATION["shape"]["weights"]), PARAMS.shape_weights)
        self.assertEqual(config.CALIBRATION["motion"]["still_speed"], PARAMS.still_speed)


if __name__ == "__main__":
    unittest.main()
