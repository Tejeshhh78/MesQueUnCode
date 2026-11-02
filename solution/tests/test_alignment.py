"""Frame alignment: ATD frame_id vs video frame, per-half offsets, graded halves."""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import games  # noqa: E402
from solution import loaders  # noqa: E402


class HalfOffsetTest(unittest.TestCase):
    def test_graded_halves_are_the_published_blocks(self):
        self.assertEqual(games.half_video_blocks("BAR-ATH"),
                         [(1, 11185, 90651), (2, 110808, 190197)])

    def test_each_half_uses_its_own_offset(self):
        # The halves differ by one frame; a single offset would be wrong for one of them.
        self.assertEqual(games.video_to_id("BAR-ATH", 11185), 11185 + 1702057)
        self.assertEqual(games.video_to_id("BAR-ATH", 110808), 110808 + 1702058)
        for frame in (11185, 50000, 90651, 110808, 150000, 190197):
            self.assertEqual(games.id_to_video("BAR-ATH", games.video_to_id("BAR-ATH", frame)),
                             frame)

    def test_halftime_and_outside_are_not_in_a_half(self):
        for frame in (0, 11184, 90652, 100000, 110807, 190198, 199318):
            self.assertIsNone(games.half_of_video_frame("BAR-ATH", frame))

    def test_graded_mask_counts_158857_frames(self):
        mask = loaders.graded_mask(199319, games.half_video_blocks("BAR-ATH"))
        self.assertEqual(int(mask.sum()), 158857)
        self.assertTrue(mask[11185] and mask[90651] and mask[110808] and mask[190197])
        self.assertFalse(mask[11184] or mask[90652] or mask[110807] or mask[190198])


class AtdTimelineTest(unittest.TestCase):
    RAW = ("1:1.0,0.5,2.0;NaN,NaN,NaN:0.5,0.5,NaN\n"
           "2:0.9,0.5,2.1;0.2,0.2,1.0:NaN,NaN,NaN\n"
           "3:0.8,0.5,2.2;0.3,0.2,1.1:0.6,0.4,9.0\n")

    def test_parse_raw_atd(self):
        frame_ids, players, ball = loaders.parse_raw_atd(self.RAW, n_slots=2)
        np.testing.assert_array_equal(frame_ids, [1, 2, 3])
        self.assertEqual(players.shape, (3, 2, 3))
        self.assertTrue(np.isnan(players[0, 1]).all())
        self.assertTrue(np.isnan(ball[1]).all())
        self.assertAlmostEqual(float(ball[2, 2]), 9.0, places=5)

    def test_wrong_slot_count_is_an_error(self):
        with self.assertRaises(ValueError):
            loaders.parse_raw_atd(self.RAW, n_slots=3)

    def test_frame_id_one_is_video_frame_zero(self):
        frame_ids, players, ball = loaders.parse_raw_atd(self.RAW, n_slots=2)
        out_players, out_ball = loaders._to_video_timeline(frame_ids, players, ball, n_frames=5)
        self.assertAlmostEqual(float(out_ball[0, 0]), 0.5, places=5)   # frame_id 1
        self.assertTrue(np.isnan(out_ball[1, 0]))                      # frame_id 2, no ball
        self.assertAlmostEqual(float(out_ball[2, 0]), 0.6, places=5)   # frame_id 3
        self.assertTrue(np.isnan(out_players[3:]).all())               # no record -> NaN

    def test_frame_ids_beyond_the_video_are_rejected(self):
        frame_ids, players, ball = loaders.parse_raw_atd(self.RAW, n_slots=2)
        with self.assertRaises(ValueError):
            loaders._to_video_timeline(frame_ids, players, ball, n_frames=2)


if __name__ == "__main__":
    unittest.main()
