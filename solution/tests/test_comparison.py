"""Prevent saved incumbent output from being reported as a different profile."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from solution.code import compare_algorithms


class ComparisonProfileTests(unittest.TestCase):
    def test_quality_only_rejects_wrong_profile_before_scoring(self):
        for record in ({}, {"solution_profile": "incumbent"}):
            with self.subTest(record=record), tempfile.TemporaryDirectory() as folder:
                runtime = Path(folder) / "runtime"
                runtime.mkdir()
                (runtime / "timings.json").write_text(json.dumps(record), encoding="utf-8")
                argv = ["compare_algorithms", "--quality-only", "--profile", "responsive",
                        "--output-dir", folder]
                error = io.StringIO()
                with mock.patch("sys.argv", argv), contextlib.redirect_stderr(error), \
                        mock.patch.object(compare_algorithms, "score_predictions") as scorer:
                    with self.assertRaises(SystemExit) as raised:
                        compare_algorithms.main()
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("different profile", error.getvalue())
                scorer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
