"""End-to-end on the gated match bundle. Skipped, with the reason, when the data is not present."""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from solution import config, loaders  # noqa: E402
from solution.run_solution import predict  # noqa: E402

try:
    loaders.check_data_available()
    MISSING = None
except loaders.DataUnavailableError:
    MISSING = "gated match data not found (see docs/DATA.md and MTAH_DATA_DIR)"


@unittest.skipIf(MISSING, MISSING)
class RealMatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.match = loaders.load_match()
        cls.labels = predict(cls.match, config.DEFAULT_PARAMS)

    def test_tracking_is_on_the_tactical_timeline(self):
        self.assertEqual(self.match.tracking.n_frames, 199319)
        self.assertEqual(self.match.tracking.players.shape[1], 46)
        self.assertEqual((self.match.tracking.home_name, self.match.tracking.away_name),
                         ("Barcelona", "Atletic Club"))

    def test_output_passes_the_official_validation(self):
        table = {"frame": np.arange(len(self.labels)), "label": self.labels}
        import pandas as pd
        self.assertIsNone(pm.validate_submission(pd.DataFrame(table), 199319))

    def test_second_half_opens_dead_until_the_kickoff(self):
        # FAQ.md: graded from 110808, kick-off at 113742; everything before it is D.
        self.assertTrue(np.all(self.labels[110808:113700] == "D"))

    def test_prediction_is_deterministic(self):
        again = predict(self.match, config.DEFAULT_PARAMS)
        np.testing.assert_array_equal(again, self.labels)


if __name__ == "__main__":
    unittest.main()
