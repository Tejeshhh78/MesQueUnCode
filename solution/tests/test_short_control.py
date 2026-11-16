"""Decoder optimum and physical safeguards for the short-control experiment."""

import itertools
import unittest

import numpy as np

from solution.code.possession import model


class ShortControlTests(unittest.TestCase):
    def test_dynamic_decoder_matches_exhaustive_optimum_with_minimum_duration(self):
        rng = np.random.default_rng(707)
        for minimum in (1, 2, 3):
            scores = rng.normal(size=(6, 3))
            costs = rng.uniform(.1, 2., size=(6, 3, 3))
            for frame in range(6):
                np.fill_diagonal(costs[frame], 0.)
            path = model.decode(scores, costs, minimum)
            def value(candidate):
                return sum(scores[t, team] for t, team in enumerate(candidate)) - sum(
                    costs[t, candidate[t-1], candidate[t]] for t in range(1, 6) if candidate[t] != candidate[t-1])
            possible = []
            for candidate in itertools.product(range(3), repeat=6):
                breaks = [0] + [i for i in range(1, 6) if candidate[i] != candidate[i-1]] + [6]
                if all(b - a >= minimum for a, b in zip(breaks[1:-2], breaks[2:-1])):
                    possible.append(value(candidate))
            self.assertAlmostEqual(value(path), max(possible), places=10)

    def test_static_and_repeated_dynamic_costs_produce_identical_paths(self):
        rng = np.random.default_rng(7)
        scores = rng.normal(size=(200, 3))
        cost = np.full((3, 3), 12.); np.fill_diagonal(cost, 0.)
        for minimum in (1, 8):
            np.testing.assert_array_equal(model.decode(scores, cost, minimum),
                                          model.decode(scores, np.tile(cost, (200, 1, 1)), minimum))
        for costs in (np.zeros((199, 3, 3)), np.full((200, 3, 3), np.nan)):
            with self.assertRaises(ValueError):
                model.decode(scores, costs)

    def inputs(self):
        n = 101
        t = (np.arange(n) - 50) / 25
        ball = np.column_stack([.5 + 5 * np.abs(t)/105, np.full(n, .5), np.zeros(n)])
        distance = np.tile([4., .2], (n, 1))
        scores = np.tile([-2., 0., -2.], (n, 1))
        cost = np.full((3, 3), 24.); cost[0,1] = cost[1,0] = 12.; np.fill_diagonal(cost, 0.)
        return ball, distance, scores, cost

    def test_discount_requires_contact_local_advantage_and_stable_team(self):
        ball, distance, scores, base = self.inputs()
        discounted = model.contact_switch_costs(ball, distance, scores, (105, 68), 25, base)
        self.assertLess(discounted[50, 0, 1], 12.)
        self.assertGreaterEqual(discounted.min(), 0.)
        np.testing.assert_array_equal(discounted[:, 2, :], np.tile(base[2], (101, 1)))
        np.testing.assert_array_equal(discounted[:, :, 2], np.tile(base[:, 2], (101, 1)))
        # A constant-velocity pass supplies no action cue even when team scores are strong.
        straight = ball.copy(); straight[:, 0] = .3 + np.arange(101)*5/(25*105)
        plain = model.contact_switch_costs(straight, distance, scores, (105, 68), 25, base)
        np.testing.assert_allclose(plain, np.broadcast_to(base, plain.shape))
        uncertain = model.contact_switch_costs(ball, distance, np.zeros_like(scores), (105, 68), 25, base)
        np.testing.assert_array_equal(uncertain, np.broadcast_to(base, uncertain.shape))

    def test_missing_ball_and_impossible_jump_cannot_support_a_discount(self):
        ball, distance, scores, base = self.inputs()
        ball[48:53] = np.nan
        cost = model.contact_switch_costs(ball, distance, scores, (105, 68), 25, base)
        np.testing.assert_array_equal(cost[48:53], np.broadcast_to(base, (5, 3, 3)))
        ball, distance, scores, base = self.inputs()
        ball[50, 0] += 10
        cost = model.contact_switch_costs(ball, distance, scores, (105, 68), 25, base)
        np.testing.assert_array_equal(cost, np.broadcast_to(base, cost.shape))


if __name__ == "__main__":
    unittest.main()
