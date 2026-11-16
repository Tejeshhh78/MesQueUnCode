"""Physical invariants and missing-observation regressions for experiment 006."""

import tempfile
import unittest
from pathlib import Path
from dataclasses import replace

import numpy as np

from solution.code.possession.data import Features, read_tracking
from solution.code.possession import model, motion
from solution.tests import test_idsse
from solution.code.import_idsse import convert_match


class ImprovementTests(unittest.TestCase):
    def test_ball_gap_stress_changes_only_deleted_observations_and_is_deterministic(self):
        from solution.code.independent_evaluation import corrupt_observations
        from solution.tests.test_control_profile import control_features
        features = control_features(1000)
        features.position_player_speed = features.player_speed.copy()
        one = corrupt_observations(features, (105, 68), 606, gaps_only=True)
        two = corrupt_observations(features, (105, 68), 606, gaps_only=True)
        missing = ~np.isfinite(one.ball).all(axis=1)
        self.assertTrue(missing.any() and (~missing).any())
        for name in ("ball", "distance", "player_speed", "position_player_speed"):
            actual = getattr(one, name)
            self.assertTrue(np.isnan(actual[missing]).all())
            np.testing.assert_array_equal(actual[~missing], getattr(features, name)[~missing])
            np.testing.assert_array_equal(actual, getattr(two, name))
        np.testing.assert_array_equal(one.mean_speed, features.mean_speed)
        self.assertTrue(np.isfinite(features.ball).all())

    def test_atd_position_features_do_not_depend_on_stream_chunking_or_provider_speed(self):
        n = 100
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta, raw = root / "meta.xml", root / "raw.txt"
            meta.write_text('<root><Team id="h"/><Team id="a"/>'
                            '<Player teamId="h"/><Player teamId="a"/></root>')
            def write(speed):
                raw.write_text("".join(f"{i+1}:{.1+i*.0008},.5,{speed};.8,.5,{speed}:.2,.5,0\n"
                                       for i in range(n)))
            write(999)
            one = read_tracking(meta, raw, n, (100, 60), block_size=1, position_motion=True)
            write(0)
            many = read_tracking(meta, raw, n, (100, 60), block_size=31, position_motion=True)
        for name in ("mean_speed", "fast_fraction", "position_player_speed"):
            np.testing.assert_array_equal(getattr(one, name), getattr(many, name))
        # Normalised coordinates are stored as float32, as in the actual ATD loader.
        np.testing.assert_allclose(one.position_player_speed[10:-10, 0], 2., atol=2e-5)

    def test_all_profiles_preserve_team_symmetry_and_half_resets(self):
        from solution.tests.test_control_profile import control_features
        features = control_features(300)
        features.position_player_speed = features.player_speed.copy()
        features.distance[170:] = [8., .5]
        features.ball[70:100] = np.nan
        features.distance[70:100] = np.nan
        blocks = [(1, 10, 129), (2, 170, 289)]
        swapped = replace(features, distance=features.distance[:, ::-1],
                          player_speed=features.player_speed[:, ::-1],
                          position_player_speed=features.position_player_speed[:, ::-1])
        for profile in model.PROFILES:
            with self.subTest(profile=profile):
                labels, _ = model.predict(features, [], blocks, (105, 68), 25, profile=profile)
                mirror, _ = model.predict(swapped, [], blocks, (105, 68), 25, profile=profile)
                np.testing.assert_array_equal(mirror, np.where(labels == "H", "A", np.where(labels == "A", "H", labels)))
                np.testing.assert_array_equal(labels[130:170], "X")
                np.testing.assert_array_equal(labels[170:290], "A")

    def test_contact_detects_velocity_change_but_not_constant_velocity(self):
        n = 101
        t = (np.arange(n) - 50) / 25
        straight = np.column_stack([5 * t, np.zeros(n)])
        bounced = np.column_stack([5 * np.abs(t), np.zeros(n)])
        observed = np.ones(n, dtype=bool)
        self.assertLess(model.contact_strength(straight, observed, 25).max(), 1e-12)
        self.assertAlmostEqual(model.contact_strength(bounced, observed, 25)[50], 1.)
        observed[48] = False
        self.assertEqual(model.contact_strength(bounced, observed, 25)[50], 0.)
        bounced[48, 0] = 100
        self.assertEqual(model.contact_strength(bounced, np.ones(n, dtype=bool), 25)[50], 0.)

    def test_gap_memory_decays_without_inventing_an_owner(self):
        n = 200
        distance = np.tile([0.2, 5.], (n, 1))
        valid = np.ones(n, dtype=bool)
        valid[50:] = False
        distance[50:] = np.nan
        args = (distance, np.ones((n, 2)), np.ones(n), valid, np.zeros(n, dtype=bool), 25)
        baseline = model.controlled_geometry(*args, model.Config())
        retained = model.controlled_geometry(*args, model.profile_config("gap_memory"))
        self.assertEqual(baseline[55], 0.)
        self.assertGreater(retained[55], retained[100])
        self.assertGreater(retained[100], retained[199])
        unobserved = model.controlled_geometry(np.full((n, 2), np.nan), np.ones((n, 2)),
                                              np.ones(n), np.zeros(n, bool), np.zeros(n, bool),
                                              25, model.profile_config("gap_memory"))
        np.testing.assert_array_equal(unobserved, 0.)

    def test_gap_memory_clears_faster_for_dead_like_frames(self):
        n = 150
        distance = np.tile([0.2, 5.], (n, 1))
        valid = np.arange(n) < 50
        distance[~valid] = np.nan
        args = (distance, np.ones((n, 2)), np.ones(n), valid)
        live = model.controlled_geometry(*args, np.zeros(n, bool), 25, model.profile_config("gap_memory"))
        dead = model.controlled_geometry(*args, ~valid, 25, model.profile_config("gap_memory"))
        self.assertLess(dead[80], live[80] / 5)

    def test_unknown_speed_is_weaker_than_observed_slow_ball(self):
        n = 150
        args = (np.tile([0.2, 5.], (n, 1)), np.ones((n, 2)))
        valid, dead = np.ones(n, bool), np.zeros(n, bool)
        config = model.profile_config("speed_guard")
        unknown = model.controlled_geometry(*args, np.full(n, np.nan), valid, dead, 25, config)
        slow = model.controlled_geometry(*args, np.ones(n), valid, dead, 25, config)
        self.assertLess(unknown[60], slow[60])
        self.assertTrue(np.isfinite(unknown).all())

    def test_position_speeds_are_translation_invariant_and_reject_jumps(self):
        t = np.arange(100)[:, None] / 25
        players = np.stack([2 * t + np.arange(2), np.zeros((100, 2))], axis=2)
        speed = motion.position_speeds(players, 25)
        np.testing.assert_allclose(speed[5:-5], 2)
        np.testing.assert_allclose(speed, motion.position_speeds(players + [50, -20], 25), equal_nan=True)
        players[50] += 100
        self.assertTrue(np.isnan(motion.position_speeds(players, 25)[48]).all())

    def test_motion_differences_never_cross_periods(self):
        n = 100
        players = np.zeros((n, 2, 2))
        players[50:] = 50
        features = Features(np.full((n, 3), .5), np.ones((n, 2)), np.ones((n, 2)),
                            np.ones(n), np.ones(n), np.full(n, 2), ("H", "A"))
        motion.apply_position_motion(features, players, [0, 1], (100, 60), 25,
                                     [(1, 0, 49), (2, 50, 99)])
        self.assertTrue(np.isnan(features.mean_speed[49:51]).all())
        self.assertTrue(np.isnan(features.position_player_speed[49:51]).all())
        self.assertEqual(features.mean_speed[25], 0.)

    def test_position_profile_requires_explicit_position_features(self):
        n = 100
        features = Features(np.full((n, 3), .5), np.ones((n, 2)), np.ones((n, 2)),
                            np.ones(n), np.ones(n), np.full(n, 2), ("H", "A"))
        with self.assertRaisesRegex(ValueError, "position_player_speed"):
            model.predict(features, [], [(1, 0, n - 1)], (100, 60), 25, profile="improved")

    def test_position_import_is_independent_of_reference_and_provider_speed(self):
        fixture = test_idsse.IDSSEImportTests()
        with tempfile.TemporaryDirectory() as directory:
            info, raw = fixture.make_files(directory)
            # Lengthen each period to support motion differentiation.
            import xml.etree.ElementTree as ET
            from datetime import datetime, timedelta
            tree = ET.parse(raw)
            for group in tree.iter("FrameSet"):
                sample = dict(group[0].attrib)
                first = int(sample["N"])
                origin = datetime.fromisoformat(sample["T"])
                for child in list(group):
                    group.remove(child)
                for i in range(60):
                    attrs = dict(sample, N=str(first + i), T=(origin + timedelta(seconds=i/25)).isoformat(),
                                 X=str(float(sample["X"]) + 2*i/25))
                    ET.SubElement(group, "Frame", **attrs)
            tree.write(raw)
            first = convert_match(info, raw, position_motion=True)
            for frame in tree.iter("Frame"):
                frame.set("S", "999")
                frame.set("BallPossession", "2")
                frame.set("BallStatus", "0")
            tree.write(raw)
            second = convert_match(info, raw, position_motion=True)
        for name in ("mean_speed", "fast_fraction", "position_player_speed", "distance"):
            np.testing.assert_allclose(first[1][name], second[1][name], equal_nan=True)
        self.assertFalse(np.array_equal(first[2], second[2]))
        np.testing.assert_allclose(first[1]["mean_speed"][15:45], 2., atol=1e-5)


if __name__ == "__main__":
    unittest.main()
