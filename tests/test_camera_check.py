import unittest

import numpy as np

from possession.camera_check import evaluate


class CameraCheckTests(unittest.TestCase):
    def test_offset_mirror_missing_ball_and_cut(self):
        rng = np.random.default_rng(23)
        fps = 5
        tactical = rng.normal(0, 7, (120, 4))
        truth = rng.uniform(-10, 10, (120, 2))
        broadcast = tactical[10:110].copy() * [-1, 1, 1, 1]
        broadcast_ball = truth[10:110].copy() * [-1, 1]
        broadcast[50:] = np.nan
        truth[15:20] = np.nan
        result = evaluate((tactical, truth), (broadcast, broadcast_ball), fps, [(1, 0, 119)])
        self.assertEqual(result["candidate_gap_frames"], 5)
        self.assertEqual(result["overlap_check_frames"], 45)
        self.assertEqual(result["p90_error_m"], 0)

    def test_constant_geometry_is_ambiguous(self):
        shape = np.zeros((120, 4))
        ball = np.zeros((120, 2))
        result = evaluate((shape, ball), (shape, ball), 5, [(1, 0, 119)])
        self.assertEqual(result["overlap_check_frames"], 0)
        self.assertIsNone(result["median_error_m"])
