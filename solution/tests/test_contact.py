"""Invented ball-action cases; no competition observations or target labels."""

import copy
from dataclasses import replace
import unittest

import numpy as np

from solution.code.possession import model
from solution.tests.test_control_profile import control_features
from solution.code.independent_evaluation import corrupt_observations


PITCH = (105., 68.)


def turn_ball(n=100):
    velocity = np.tile([8., 0.], (n, 1))
    velocity[n // 2:] = [0., 8.]
    return .25 + np.cumsum(velocity / 25, axis=0) / np.asarray(PITCH)


class ContactTests(unittest.TestCase):
    def test_observation_stress_is_deterministic_and_does_not_modify_inputs(self):
        f = control_features(4000)
        original = f.ball.copy()
        a = corrupt_observations(f, PITCH, 505)
        b = corrupt_observations(f, PITCH, 505)
        np.testing.assert_array_equal(a.ball, b.ball)
        np.testing.assert_array_equal(f.ball, original)
        missing = ~np.isfinite(a.ball).all(axis=1)
        self.assertGreater(int(missing.sum()), 0)
        self.assertTrue(np.isnan(a.distance[missing]).all())
        np.testing.assert_array_equal(a.distance[~missing], f.distance[~missing])

    def test_pass_flight_and_stationary_live_dead_counterexamples(self):
        f = control_features(150)
        f.ball[:, 0] = .15 + np.arange(150) * 15 / (25 * 105)
        f.distance[60:72] = [5.5, 2.2]
        labels, _ = model.predict(f, [], [(1, 0, 149)], PITCH, 25, profile='contact')
        np.testing.assert_array_equal(labels, 'H')
        for dead in (False, True):
            f = control_features(400)
            f.ball[130:260, :2] = f.ball[130, :2]
            f.mean_speed[:] = 3.5
            if dead:
                f.mean_speed[130:260] = .3
            labels, _ = model.predict(f, [], [(1, 0, 399)], PITCH, 25, profile='contact')
            np.testing.assert_array_equal(labels[160:230], 'D' if dead else 'H')

    def test_constant_motion_and_stationary_ball_have_no_action(self):
        f = control_features(100)
        for ball in (f.ball[:, :2], np.full((100, 2), .5)):
            np.testing.assert_array_equal(model.ball_contact_strength(ball, np.ones(100, bool), PITCH, 25), 0)

    def test_turn_is_bounded_reflection_invariant_and_no_gap_is_interpolated(self):
        ball = turn_ball()
        observed = np.ones(100, bool)
        strength = model.ball_contact_strength(ball, observed, PITCH, 25)
        self.assertGreater(strength[49], .9)
        self.assertTrue(np.all((strength >= 0) & (strength <= 1)))
        np.testing.assert_allclose(strength, model.ball_contact_strength(1-ball, observed, PITCH, 25))
        observed[49] = False
        strength = model.ball_contact_strength(ball, observed, PITCH, 25)
        np.testing.assert_array_equal(strength[46:53], 0)
        ball[49] = np.nan
        np.testing.assert_array_equal(model.ball_contact_strength(ball, np.ones(100, bool), PITCH, 25)[46:53], 0)

    def test_isolated_implausible_spike_is_not_a_touch(self):
        ball = np.full((40, 2), .5)
        ball[20, 0] = .9
        np.testing.assert_array_equal(model.ball_contact_strength(ball, np.ones(40, bool), PITCH, 25), 0)

    def test_close_turn_strengthens_bounded_evidence_but_not_dead_like_regions(self):
        distance = np.tile([1.2, .5], (80, 1))
        args = (distance, np.full((80, 2), 3.), np.full(80, 6.), np.ones(80, bool))
        config = model.profile_config('controlled')
        for dead in (False, True):
            mask = np.full(80, dead)
            original = model.controlled_geometry(*args, mask, 25, config)
            modified = model.controlled_geometry(*args, mask, 25, config, contact_strength=np.ones(80))
            if dead:
                np.testing.assert_array_equal(original, modified)
            else:
                self.assertLess(modified[40], original[40])
                self.assertTrue(np.all(np.abs(modified) <= 2.5))
        # Equally close players cannot be distinguished by a ball action alone.
        ambiguous = (np.full((80, 2), .4), *args[1:])
        evidence = model.controlled_geometry(*ambiguous, np.zeros(80, bool), 25, config,
                                             contact_strength=np.ones(80))
        np.testing.assert_array_equal(evidence, 0)

    def test_controlled_without_actions_and_dead_evidence_are_preserved(self):
        f = control_features(300)
        f.distance[100:140] = [8., .5]
        original, modified = {}, {}
        a, _ = model.predict(f, [], [(1, 0, 299)], PITCH, 25, profile='controlled', trace=original)
        b, _ = model.predict(f, [], [(1, 0, 299)], PITCH, 25, profile='contact', trace=modified)
        np.testing.assert_array_equal(a, b)
        for key in original:
            np.testing.assert_array_equal(original[key], modified[key])
        f.ball[:, :2] = turn_ball(300)
        model.predict(f, [], [(1, 0, 299)], PITCH, 25, profile='controlled', trace=original)
        model.predict(f, [], [(1, 0, 299)], PITCH, 25, profile='contact', trace=modified)
        for key in ('alive', 'moving', 'stationary', 'missing', 'tag_team', 'status_tags'):
            np.testing.assert_array_equal(original[key], modified[key])
        np.testing.assert_array_equal(original['scores'][:, 2], modified['scores'][:, 2])
        self.assertEqual(model.profile_config('controlled'), model.profile_config('contact'))

    def test_missing_ball_turnover_period_reset_and_team_swap(self):
        f = control_features(300)
        f.ball[100:130] = np.nan
        f.distance[100:130] = np.nan
        f.distance[130:] = [8., .5]
        labels, _ = model.predict(f, [], [(1, 0, 299)], PITCH, 25, profile='contact')
        np.testing.assert_array_equal(labels[:100], 'H')
        np.testing.assert_array_equal(labels[140:], 'A')
        swapped = replace(f, distance=f.distance[:, ::-1], player_speed=f.player_speed[:, ::-1])
        other, _ = model.predict(swapped, [], [(1, 0, 299)], PITCH, 25, profile='contact')
        np.testing.assert_array_equal(other, np.where(labels == 'H', 'A', np.where(labels == 'A', 'H', labels)))
        before = copy.deepcopy(f)
        model.predict(f, [], [(1, 0, 99), (2, 130, 299)], PITCH, 25, profile='contact')
        for field in ('ball', 'distance', 'player_speed'):
            np.testing.assert_array_equal(getattr(f, field), getattr(before, field))
        # A turn at a period join must not supply cross-period action evidence.
        ball = turn_ball()
        left = model.ball_contact_strength(ball[:50], np.ones(50, bool), PITCH, 25)
        right = model.ball_contact_strength(ball[50:], np.ones(50, bool), PITCH, 25)
        np.testing.assert_array_equal(left, 0)
        np.testing.assert_array_equal(right, 0)


if __name__ == '__main__':
    unittest.main()
