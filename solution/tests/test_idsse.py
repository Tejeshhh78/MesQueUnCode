"""Synthetic checks for native DFL import and separation of reference labels."""

from datetime import datetime, timedelta, timezone
import csv
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from solution.code.import_idsse import MATCHES, convert_match, import_dataset, reference_label


class IDSSEImportTests(unittest.TestCase):
    def make_files(self, directory, *, alternate_labels=False, duplicate=False, fps=25):
        directory = Path(directory)
        info = directory / "info.xml"
        info.write_text('''<PutDataRequest><MatchInformation>
            <General MatchId="DFL-MAT-J03WMX" HomeTeamId="HOME" GuestTeamId="AWAY"
            HomeTeamName="Home" GuestTeamName="Away" KickoffTime="2023-01-01T00:00:00Z"/>
            <Environment PitchX="100" PitchY="60"/>
            </MatchInformation></PutDataRequest>''', encoding="utf-8")
        root = ET.Element("PutDataRequest")
        positions = ET.SubElement(root, "Positions")
        ET.SubElement(ET.SubElement(positions, "MetaData", MatchId="DFL-MAT-J03WMX"),
                      "PitchSize", X="100", Y="60")
        for half, start in (("firstHalf", 100), ("secondHalf", 1000)):
            for team, x, y, speed in (("HOME", 3, 4, 7.2), ("AWAY", 10, 0, 14.4),
                                      ("referee", 1, 0, 36), ("BALL", 0, 0, 18)):
                group = ET.SubElement(positions, "FrameSet", GameSection=half,
                                      MatchId="DFL-MAT-J03WMX", TeamId=team, PersonId=team)
                for i in range(4):
                    attrs = {"N": str(start + (0 if duplicate and i == 1 and team == "HOME" else i)),
                             "T": (datetime(2023, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=i / fps)).isoformat(),
                             "X": str(x), "Y": str(y), "S": str(speed)}
                    if team == "BALL":
                        attrs.update(BallStatus="1" if alternate_labels else ["1", "1", "0", "1"][i],
                                     BallPossession="2" if alternate_labels else ["1", "2", "0", "0"][i], Z="999")
                    ET.SubElement(group, "Frame", **attrs)
        raw = directory / "positions.xml"
        ET.ElementTree(root).write(raw, encoding="utf-8", xml_declaration=True)
        return info, raw

    def test_units_coordinates_labels_and_half_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata, features, labels, blocks = convert_match(*self.make_files(directory))
        self.assertEqual(metadata["teams"], ("Home", "Away"))
        self.assertEqual(blocks, [{"period": 1, "start": 0, "stop": 4},
                                  {"period": 2, "start": 4, "stop": 8}])
        np.testing.assert_array_equal(labels, list("HADXHADX"))
        np.testing.assert_allclose(features["ball"], np.tile([.5, .5, 5.], (8, 1)))
        np.testing.assert_allclose(features["distance"], np.tile([5., 10.], (8, 1)))
        np.testing.assert_allclose(features["player_speed"], np.tile([2., 4.], (8, 1)))
        np.testing.assert_allclose(features["mean_speed"], 3.)
        np.testing.assert_allclose(features["fast_fraction"], .5)
        np.testing.assert_array_equal(features["player_count"], 2)

    def test_reference_changes_cannot_change_features_or_periods(self):
        with tempfile.TemporaryDirectory() as directory:
            first = convert_match(*self.make_files(directory))
            second = convert_match(*self.make_files(directory, alternate_labels=True))
        self.assertFalse(np.array_equal(first[2], second[2]))
        self.assertEqual(first[3], second[3])
        for name in first[1]:
            np.testing.assert_array_equal(first[1][name], second[1][name])

    def test_unknown_status_and_possession_are_not_dead_ball(self):
        self.assertEqual(reference_label({}), "X")
        self.assertEqual(reference_label({"BallStatus": "1", "BallPossession": "0"}), "X")
        self.assertEqual(reference_label({"BallStatus": "unknown", "BallPossession": "1"}), "X")
        self.assertEqual(reference_label({"BallStatus": "0", "BallPossession": "2"}), "D")

    def test_reject_non_native_frame_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "25 Hz"):
                convert_match(*self.make_files(directory, fps=10))

    def test_reject_duplicate_player_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Duplicate or unordered"):
                convert_match(*self.make_files(directory, duplicate=True))

    def test_reject_mismatched_match_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            info, raw = self.make_files(directory)
            raw.write_text(raw.read_text().replace("DFL-MAT-J03WMX", "DFL-MAT-J03WN1"))
            with self.assertRaisesRegex(ValueError, "identit"):
                convert_match(info, raw)

    def test_manifest_export_uses_zero_based_frame_contract_and_disjoint_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            info, raw = self.make_files(directory)
            for match in ("J03WMX", "J03WN1"):
                suffix = f"DFL-COM-{MATCHES[match][0]}_DFL-MAT-{match}.xml"
                (root / f"DFL_02_01_matchinformation_{suffix}").write_text(
                    info.read_text().replace("J03WMX", match), encoding="utf-8")
                (root / f"DFL_04_03_positions_raw_observed_{suffix}").write_text(
                    raw.read_text().replace("J03WMX", match), encoding="utf-8")
            manifest_path = import_dataset(root, root / "out", ["J03WMX"], ["J03WN1"])
            manifest = json.loads(manifest_path.read_text())
            with (root / "out/J03WMX_labels.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(row["frame"]) for row in rows], list(range(8)))
            self.assertEqual([match["split"] for match in manifest["matches"]], ["development", "final_test"])
            self.assertTrue(manifest["protocol"]["final_test_locked"])
            with self.assertRaisesRegex(ValueError, "disjoint"):
                import_dataset(root, root / "bad", ["J03WMX"], ["J03WMX"])


if __name__ == "__main__":
    unittest.main()
