"""Synthetic tests without labels or scenes from the competition match."""

import itertools
from pathlib import Path
import tempfile
import unittest

import numpy as np

from evaluation import possession_metrics as pm
from solution.code.possession.__main__ import write_submission
from solution.code.possession.data import Features, read_events, read_tracking
from solution.code.possession.model import (Config, activity_evidence, decode, fill_short_gaps,
                              predict, rolling_mean, tagging_evidence, team_side)


def synthetic_features(n=400):
    # Both teams are running; the ball moves at 5 m/s near the home team.
    ball = np.zeros((n, 3))
    ball[:, 0] = 0.15 + np.arange(n) / 25 * 5 / 105
    ball[:, 1] = 0.5
    ball[:, 2] = 5
    return Features(ball, np.tile([0.5, 8.0], (n, 1)), np.full((n, 2), 3.),
                    np.full(n, 3.), np.full(n, 0.6), np.full(n, 22),
                    ("FC Barcelona", "Athletic Club"))


class DataTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.meta = self.root / "meta.xml"
        self.raw = self.root / "raw.txt"
        self.meta.write_text('<root><Team id="h"><Name>Home</Name></Team>'
                             '<Team id="a"><Name>Away</Name></Team>'
                             '<Player teamId="h"/><Player teamId="a"/></root>')

    def test_team_mapping_metres_missing_ball_and_one_based_frame(self):
        self.raw.write_text("1:0.5,0.5,2;0.51,0.48,3:0.5,0.49,0\n"
                            "2:NaN,NaN,NaN;0.7,0.8,1:NaN,NaN,NaN\n")
        f = read_tracking(self.meta, self.raw, 2, (100, 50))
        np.testing.assert_allclose(f.distance[0], [0.5, np.sqrt(1.25)])
        self.assertTrue(np.isnan(f.distance[1]).all())
        self.assertEqual(f.player_count.tolist(), [2, 1])

    def test_bad_frames_truncation_and_bad_numbers_rejected(self):
        for content, count in [("2:0,0,1;1,1,1:0,0,0\n", 1),
                                ("1:0,0,1;1,1,1:0,0,0\n", 2),
                                ("1:0,0,1:0,0,0\n", 1),
                                ("1:0,0,bad;1,1,1:0,0,0\n", 1),
                                ("1:0,0,1;1,1,1:inf,0,0\n", 1)]:
            with self.subTest(content=content):
                self.raw.write_text(content)
                with self.assertRaises(ValueError):
                    read_tracking(self.meta, self.raw, count, (100, 50))

    def test_unknown_track_team_rejected(self):
        self.meta.write_text('<root><Team id="h"/><Team id="a"/><Player teamId="other"/></root>')
        self.raw.write_text("1:0,0,0:0,0,0\n")
        with self.assertRaises(ValueError):
            read_tracking(self.meta, self.raw, 1, (100, 50))

    def test_block_boundaries_do_not_change_features(self):
        rng = np.random.default_rng(83)
        records = rng.uniform(0, 1, size=(19, 3, 3))
        records[3:8, 0] = np.nan
        records[11, :, 2] = np.nan
        records[13:16, -1] = np.nan
        self.raw.write_text("".join(
            f"{i + 1}:" + ";".join(",".join(map(str, p)) for p in record[:2])
            + ":" + ",".join(map(str, record[2])) + "\n"
            for i, record in enumerate(records)))
        single = read_tracking(self.meta, self.raw, len(records), (100, 50), block_size=1)
        for size in (2, 7, 4096):
            block = read_tracking(self.meta, self.raw, len(records), (100, 50), block_size=size)
            for field in ("ball", "distance", "player_speed", "mean_speed", "fast_fraction", "player_count"):
                np.testing.assert_array_equal(getattr(single, field), getattr(block, field))

    def test_partial_numeric_tokens_and_empty_fields_rejected(self):
        for ball in ("0,0,1junk", "0,0,1 2", "0,,1", "0,0,", "0,0,nanX"):
            with self.subTest(ball=ball):
                self.raw.write_text("1:0,0,1;1,1,1:" + ball + "\n")
                with self.assertRaises(ValueError):
                    read_tracking(self.meta, self.raw, 1, (100, 50))

    def test_tag_alignment_and_case(self):
        xml = self.root / "tags.xml"
        xml.write_text('<root><instance><start>0.04</start><end>0.12</end><code>Build Up</code>'
                        '<label><group>Team</group><text>Home</text></label></instance></root>')
        event = read_events(xml, 4, 25)[0]
        self.assertEqual((event["start"], event["stop"], event["code"]), (1, 3, "buildup"))
        self.assertEqual(event["tags"]["team"], "Home")
        with self.assertRaises(ValueError):
            read_events(xml, 1, 25)


class InferenceTests(unittest.TestCase):
    def test_rolling_edges_and_short_gaps(self):
        np.testing.assert_allclose(rolling_mean([1, np.nan, 3, 5], 3), [1, 2, 4, 4])
        points = np.array([[0, 0], [np.nan, np.nan], [2, 2], [np.nan, np.nan],
                           [np.nan, np.nan], [5, 5], [np.nan, np.nan]])
        filled = fill_short_gaps(points, 1)
        np.testing.assert_allclose(filled[1], [1, 1])
        self.assertTrue(np.isnan(filled[3:5]).all())
        self.assertTrue(np.isnan(filled[-1]).all())

    def test_decoder_equals_exhaustive_optimum(self):
        rng = np.random.default_rng(42)
        cost = np.array([[0., 1.1, 1.8], [1.1, 0., 1.8], [1.8, 1.8, 0.]])
        for k in (1, 2, 3):
            scores = rng.normal(size=(6, 3))
            path = decode(scores, cost, k)
            def objective(p):
                return sum(scores[t, p[t]] for t in range(6)) - sum(cost[p[t - 1], p[t]] for t in range(1, 6))
            candidates = []
            for candidate in itertools.product(range(3), repeat=6):
                boundaries = [0] + [i for i in range(1, 6) if candidate[i] != candidate[i - 1]] + [6]
                if all(b - a >= k for a, b in zip(boundaries[1:-2], boundaries[2:-1])):
                    candidates.append(objective(candidate))
            self.assertAlmostEqual(objective(path), max(candidates), places=10)

    def test_glance_suppressed_but_controlled_change_keeps_boundary(self):
        scores = np.full((200, 3), -5.)
        scores[:100, 0] = 0
        scores[100:, 1] = 0
        scores[40:43, 1] = 1
        cost = np.full((3, 3), 20.)
        np.fill_diagonal(cost, 0)
        path = decode(scores, cost, 8)
        np.testing.assert_array_equal(path[:100], 0)
        np.testing.assert_array_equal(path[100:], 1)

    def test_decoder_ties_preserve_defined_predecessor_order(self):
        scores = np.zeros((9, 3))
        cost = np.zeros((3, 3))
        np.testing.assert_array_equal(decode(scores, cost, 1), np.zeros(9, dtype=int))
        # The first age state wins the final tie. As a boundary state it may be
        # short; its entry predecessor is the smaller of the other team indices.
        for minimum in (2, 3):
            np.testing.assert_array_equal(decode(scores, cost, minimum), [1] * 8 + [0])

    def test_long_ball_gap_is_not_automatically_dead(self):
        f = synthetic_features()
        f.ball[100:250] = np.nan
        f.distance[100:250] = np.nan
        labels, _ = predict(f, [], [(1, 0, 399)], (105, 68), 25)
        self.assertTrue(np.all(labels == "H"))

    def test_stationary_low_activity_stoppage_and_restart(self):
        f = synthetic_features()
        f.mean_speed[130:260] = 0.3
        f.ball[130:260, :2] = f.ball[130, :2]
        labels, _ = predict(f, [], [(1, 0, 399)], (105, 68), 25)
        self.assertTrue(np.all(labels[160:230] == "D"))
        self.assertTrue(np.all(labels[280:350] == "H"))

    def test_halves_reset_and_ungraded_frames_abstain(self):
        f = synthetic_features(400)
        f.distance[210:] = [8., 0.5]
        labels, _ = predict(f, [], [(1, 10, 189), (2, 210, 389)], (105, 68), 25)
        self.assertTrue(np.all(labels[10:190] == "H"))
        self.assertTrue(np.all(labels[210:390] == "A"))
        self.assertTrue(np.all(labels[190:210] == "X"))

    def test_soft_tags_defence_inversion_unknown_teams_and_duplicate_invariance(self):
        self.assertEqual(team_side("Atletic Club", ("Barcelona", "Athletic Club")), 1)
        self.assertIsNone(team_side("N/A", ("Barcelona", "Athletic Club")))
        event = {"start": 0, "stop": 100, "code": "defendinginmiddlethird", "tags": {"team": "Away"}}
        args = (("Home", "Away"), 100, 25, np.full(100, 5.), np.full(100, 3.))
        one = tagging_evidence([event], *args)
        duplicate = tagging_evidence([event, event], *args)
        self.assertTrue(np.all(one[0][20:80] > 0))
        np.testing.assert_array_equal(one[0], duplicate[0])

    def test_activity_is_deterministic_and_missing_neutral(self):
        speed = np.r_[np.full(250, 0.4), np.full(250, 3.), np.nan]
        evidence, _ = activity_evidence(speed, np.ones(501, dtype=bool), 25)
        self.assertLess(evidence[50], 0)
        self.assertGreater(evidence[350], 0)
        self.assertEqual(evidence[-1], 0)
        np.testing.assert_array_equal(evidence, activity_evidence(speed, np.ones(501, dtype=bool), 25)[0])

    def test_synthetic_labels_score_perfectly_and_format_is_deterministic(self):
        with tempfile.TemporaryDirectory() as folder:
            labels = np.array(list("HHHHHDDDDAAAAAXX"))
            a, b = Path(folder) / "a.csv", Path(folder) / "b.csv"
            write_submission(a, labels)
            write_submission(b, labels)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            self.assertNotIn(b"\r", a.read_bytes())
            frame = pm.read_label_csv(a)
            self.assertIsNone(pm.validate_submission(frame, len(labels)))
            codes = pm.label_codes(frame)
            result = pm.evaluate(codes, codes)
            self.assertEqual(result["spell_f1_mean"], 1.)
            self.assertEqual(result["macro_f1"], 1.)


if __name__ == "__main__":
    unittest.main()
