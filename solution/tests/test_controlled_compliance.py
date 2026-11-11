"""The organisers' rule clarification, enforced for the controlled model: nothing is fitted on the
provided match; the movement regimes come from external games and are only applied."""

import json
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from solution.code.possession import model, motion  # noqa: E402


class FrozenActivityTest(unittest.TestCase):
    def test_evidence_does_not_depend_on_the_rest_of_the_match(self):
        # With a per-match fit, a 2 m/s frame would score differently in a slow match and in a
        # fast one. Frozen regimes give it the same evidence whatever surrounds it.
        mostly_slow = np.r_[np.full(900, 0.5), 2.0]
        mostly_fast = np.r_[np.full(900, 4.0), 2.0]
        slow_evidence, _ = model.activity_evidence(mostly_slow, np.ones(901, dtype=bool), 25)
        fast_evidence, _ = model.activity_evidence(mostly_fast, np.ones(901, dtype=bool), 25)
        self.assertEqual(slow_evidence[-1], fast_evidence[-1])

    def test_prediction_never_fits(self):
        def refuse(*args, **kwargs):
            raise AssertionError("fit_regimes was called at inference time")
        original = model.fit_regimes
        model.fit_regimes = refuse
        try:
            speed = np.r_[np.full(250, 0.4), np.full(250, 3.0)]
            model.activity_evidence(speed, np.ones(500, dtype=bool), 25)
        finally:
            model.fit_regimes = original

    def test_calibration_is_external(self):
        data = json.loads(motion.CALIBRATION_PATH.read_text(encoding="utf-8"))
        self.assertFalse(data["provenance"]["provided_match_used"])
        self.assertIn("metrica-sports/sample-data", data["provenance"]["data"])
        means, variances, weights = motion.load_regimes()
        self.assertEqual(len(means), 2)
        self.assertLess(means[0], means[1])
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=5)

    def test_missing_calibration_fails_clearly(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                motion.load_regimes(os.path.join(folder, "absent.json"))

    def test_calibration_without_external_provenance_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "c.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"regimes": {"means": [0, 1], "variances": [1, 1], "weights": [0.5, 0.5]},
                           "provenance": {}}, handle)
            with self.assertRaises(ValueError):
                motion.load_regimes(path)


class RobustMotionTest(unittest.TestCase):
    def test_constant_running_speed_is_recovered(self):
        t = np.arange(500)[:, None]
        players = np.stack([3.0 * t / 25.0 + np.arange(4) * 5, np.full((500, 4), 30.0)], axis=2)
        mean, fast = motion.robust_motion(players, 25)
        np.testing.assert_allclose(mean[20:-20], 3.0, atol=1e-9)
        np.testing.assert_allclose(fast[20:-20], 1.0)

    def test_positional_jitter_barely_moves_it(self):
        rng = np.random.default_rng(0)
        t = np.arange(2000)[:, None]
        players = np.stack([1.5 * t / 25.0 + np.arange(20) * 2, np.full((2000, 20), 30.0)], axis=2)
        noisy = players + rng.normal(0, 0.2, size=players.shape)
        clean_mean, _ = motion.robust_motion(players, 25)
        noisy_mean, _ = motion.robust_motion(noisy, 25)
        self.assertLess(np.nanmean(np.abs(noisy_mean - clean_mean)), 0.3)

    def test_unseen_players_are_ignored(self):
        players = np.full((100, 3, 2), np.nan)
        mean, fast = motion.robust_motion(players, 25)
        self.assertTrue(np.isnan(mean).all() and np.isnan(fast).all())


try:
    from solution import loaders
    loaders.check_data_available()
    MISSING = None
except Exception as error:  # data absent or bundle unreadable: the real-data check is skipped
    MISSING = "gated match data not available: {}".format(type(error).__name__)


@unittest.skipIf(MISSING, MISSING)
class ControlledOnRealDataTest(unittest.TestCase):
    """Automatic checks on the provided match (allowed by the rules). They report pass/fail only."""

    def test_end_to_end_output_is_valid_and_deterministic(self):
        import contextlib
        import io
        from solution.code.possession.__main__ import run
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            with contextlib.redirect_stdout(io.StringIO()):
                one = run(None, first, "MesQueUnCode")
                two = run(None, second, "MesQueUnCode")
        self.assertEqual(one["submission_sha256"], two["submission_sha256"])
        self.assertEqual(one["rows"], 199319)
        self.assertFalse(one["ground_truth_used"])
        self.assertEqual(one["model"]["activity_model"]["fitted"], "frozen from external games")


if __name__ == "__main__":
    unittest.main()
