"""Import public IDSSE/DFL matches without feeding reference fields to inference.

Source: Bassek et al. (2025), doi:10.1038/s41597-025-04505-y.
Data: Deutsche Fussball Liga, doi:10.6084/m9.figshare.28196177.v1, CC BY 4.0.
Only native 25 Hz files are accepted. No event-derived tactical tags are created.
"""

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


SOURCE = "https://doi.org/10.6084/m9.figshare.28196177.v1"
PAPER = "https://doi.org/10.1038/s41597-025-04505-y"
PERIODS = {"firstHalf": 1, "secondHalf": 2}
MATCHES = {
    "J03WMX": ("000001", 51643475, 51643514),
    "J03WN1": ("000001", 51643472, 51643517),
    "J03WOH": ("000002", 51643478, 51643520),
    "J03WOY": ("000002", 51643481, 51643523),
    "J03WPY": ("000002", 51643487, 51643526),
    "J03WQQ": ("000002", 51643484, 51643529),
    "J03WR9": ("000002", 51643490, 51643532),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_metadata(path):
    root = ET.parse(path).getroot()
    general = root.find("MatchInformation/General")
    environment = root.find("MatchInformation/Environment")
    if general is None or environment is None:
        raise ValueError("Missing DFL match metadata")
    ids = (general.get("HomeTeamId"), general.get("GuestTeamId"))
    names = (general.get("HomeTeamName"), general.get("GuestTeamName"))
    if not all(ids + names) or ids[0] == ids[1]:
        raise ValueError("Missing or duplicate DFL team identity")
    pitch = (float(environment.get("PitchX")), float(environment.get("PitchY")))
    if not np.isfinite(pitch).all() or min(pitch) <= 0:
        raise ValueError("Invalid DFL pitch dimensions")
    match_id = general.get("MatchId")
    if match_id not in {f"DFL-MAT-{key}" for key in MATCHES}:
        raise ValueError("Only the seven public IDSSE matches are supported")
    return {"match_id": match_id, "team_ids": ids, "teams": names,
            "pitch": pitch, "date": general.get("KickoffTime")}


def iter_frames(path, metadata, *, ball_only):
    """Stream frames, validating source identity and excluding referee tracks."""
    current = None
    frameset = None
    previous_frame = None
    seen_sets = set()
    for event, element in ET.iterparse(path, events=("start", "end")):
        if event == "start" and element.tag == "MetaData":
            if element.get("MatchId") != metadata["match_id"]:
                raise ValueError("Position and match metadata identities differ")
        if event == "start" and element.tag == "PitchSize":
            pitch = (float(element.get("X")), float(element.get("Y")))
            if not np.allclose(pitch, metadata["pitch"], rtol=0, atol=1e-6):
                raise ValueError("Position and match pitch dimensions differ")
        if event == "start" and element.tag == "FrameSet":
            if element.get("MatchId") != metadata["match_id"]:
                raise ValueError("Unexpected match identity in position frameset")
            period, team = element.get("GameSection"), element.get("TeamId")
            if period not in PERIODS:
                raise ValueError("Unsupported DFL game section")
            if team not in (*metadata["team_ids"], "BALL", "referee"):
                raise ValueError("Unknown DFL tracking team")
            key = (period, team, element.get("PersonId"))
            if key in seen_sets:
                raise ValueError("Duplicate DFL object frameset")
            seen_sets.add(key)
            current = (period, team) if (team == "BALL") == ball_only and team != "referee" else None
            frameset = element
            previous_frame = None
        if event == "end" and element.tag == "Frame":
            if current is not None:
                frame_number = int(element.attrib["N"])
                if previous_frame is not None and frame_number <= previous_frame:
                    raise ValueError("Duplicate or unordered DFL object frame")
                previous_frame = frame_number
                yield current, element.attrib
            element.clear()
            if frameset is not None:
                frameset.remove(element)
        elif event == "end" and element.tag == "FrameSet":
            element.clear()
            current, frameset = None, None


def number(attributes, key):
    value = attributes.get(key)
    result = float(value) if value not in (None, "") else float("nan")
    if np.isinf(result):
        raise ValueError("Infinite DFL tracking value")
    return result


def reference_label(attributes):
    """Unknown live possession remains ungraded; it is never a stoppage."""
    status = attributes.get("BallStatus")
    if status == "0":
        return "D"
    if status == "1":
        return {"1": "H", "2": "A"}.get(attributes.get("BallPossession"), "X")
    return "X"


def convert_match(metadata_path, positions_path):
    """Return geometry-only features and a separate native provider reference.

    Period boundaries use FrameSet GameSection and frame numbers, never labels.
    All source tracking observations are retained, including dead-ball frames.
    """
    metadata = read_metadata(metadata_path)
    records = {key: [] for key in PERIODS}
    for (period, _), row in iter_frames(positions_path, metadata, ball_only=True):
        records[period].append((int(row["N"]), datetime.fromisoformat(row["T"]),
                               number(row, "X"), number(row, "Y"), number(row, "S") / 3.6,
                               reference_label(row)))
    blocks, bounds = [], {}
    offset = 0
    for period, rows in records.items():
        if len(rows) < 2:
            raise ValueError("Each DFL half must contain at least two ball frames")
        first, last = rows[0][0], rows[-1][0]
        if last - first + 1 != len(rows) or any(row[0] != first + i for i, row in enumerate(rows)):
            raise ValueError("DFL ball frames must be consecutive within each half")
        origin = rows[0][1]
        if any(abs((row[1] - origin).total_seconds() - i / 25) > 0.001 for i, row in enumerate(rows)):
            raise ValueError("DFL timestamps must follow the native 25 Hz frame grid")
        bounds[period] = (first, last, offset, origin)
        blocks.append({"period": PERIODS[period], "start": offset, "stop": offset + len(rows)})
        offset += len(rows)
    n = offset
    features = {"ball": np.full((n, 3), np.nan),
                "distance": np.full((n, 2), np.nan),
                "player_speed": np.full((n, 2), np.nan),
                "mean_speed": np.full(n, np.nan),
                "fast_fraction": np.full(n, np.nan),
                "player_count": np.zeros(n, dtype=np.int16)}
    labels = np.full(n, "X", dtype="U1")
    for period, rows in records.items():
        start = bounds[period][2]
        stop = start + len(rows)
        ball = np.asarray([row[2:5] for row in rows], dtype=float)
        ball[:, :2] = ball[:, :2] / np.asarray(metadata["pitch"]) + 0.5
        features["ball"][start:stop] = ball
        labels[start:stop] = [row[5] for row in rows]
    del records
    speed_sum = np.zeros(n)
    speed_count = np.zeros(n, dtype=np.int16)
    fast_count = np.zeros(n, dtype=np.int16)
    nearest = np.full((n, 2), np.inf)
    for (period, team_id), row in iter_frames(positions_path, metadata, ball_only=False):
        first, last, start, origin = bounds[period]
        frame = int(row["N"])
        # Tracking can include pre/post-period player samples; keep the ball grid.
        if frame < first or frame > last:
            continue
        if abs((datetime.fromisoformat(row["T"]) - origin).total_seconds() - (frame - first) / 25) > 0.001:
            raise ValueError("Player and ball timestamp grids differ")
        index, team = start + frame - first, metadata["team_ids"].index(team_id)
        x, y, speed = number(row, "X"), number(row, "Y"), number(row, "S") / 3.6
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        features["player_count"][index] += 1
        if np.isfinite(speed) and 0 <= speed <= 15:
            speed_sum[index] += speed
            speed_count[index] += 1
            fast_count[index] += speed > 2
        ball_xy = (features["ball"][index, :2] - 0.5) * metadata["pitch"]
        distance = float(np.hypot(x - ball_xy[0], y - ball_xy[1]))
        if distance < nearest[index, team]:
            nearest[index, team] = distance
            features["distance"][index, team] = distance
            features["player_speed"][index, team] = speed
    valid = speed_count > 0
    features["mean_speed"][valid] = speed_sum[valid] / speed_count[valid]
    features["fast_fraction"][valid] = fast_count[valid] / speed_count[valid]
    return metadata, features, labels, blocks


def import_dataset(source_dir, output_dir, development, final_test):
    """Write a locked, whole-match split and separate feature/reference files."""
    development, final_test = list(development), list(final_test)
    chosen = development + final_test
    if not development or not final_test or len(chosen) != len(set(chosen)):
        raise ValueError("Use disjoint, nonempty development and final-test match lists")
    if any(match not in MATCHES for match in chosen):
        raise ValueError("Unknown public IDSSE match")
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Refusing to overwrite an existing split manifest")
    adapter_hash = sha256(__file__)
    manifest = {"schema_version": 1, "dataset_id": "IDSSE-2025-v1",
                "protocol": {"id": "idsse-whole-match-pilot-v1",
                             "selection_rule": "Match IDs fixed before any model scores; separate complete matches; pilot only.",
                             "declared_before_results": True, "split_unit": "match", "final_test_locked": True},
                "matches": []}
    for match in chosen:
        competition, info_id, position_id = MATCHES[match]
        suffix = f"DFL-COM-{competition}_DFL-MAT-{match}.xml"
        info = source_dir / f"DFL_02_01_matchinformation_{suffix}"
        positions = source_dir / f"DFL_04_03_positions_raw_observed_{suffix}"
        metadata, features, labels, blocks = convert_match(info, positions)
        npz_path, label_path = output_dir / f"{match}.npz", output_dir / f"{match}_labels.csv"
        if npz_path.exists() or label_path.exists():
            raise FileExistsError("Refusing to overwrite converted match files")
        np.savez_compressed(npz_path, **features)
        with label_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("frame", "label"))
            writer.writerows(enumerate(labels))
        manifest["matches"].append({
            "match_id": metadata["match_id"], "group_id": metadata["match_id"],
            "split": "development" if match in development else "final_test",
            "evidence_type": "reference", "fps": 25, "n_frames": len(labels),
            "pitch_m": list(metadata["pitch"]), "teams": list(metadata["teams"]),
            "units": {"positions": "normalised_pitch", "distances": "metres", "speeds": "metres_per_second"},
            "blocks": blocks, "inputs": {"format": "features_npz", "path": npz_path.name},
            "labels": {"path": label_path.name},
            "provenance": {
                "source": SOURCE, "license": "CC BY 4.0: https://creativecommons.org/licenses/by/4.0/",
                "organiser_access": "Public, no authentication: " + SOURCE,
                "match_identity": f"{metadata['match_id']}; {metadata['date']}; {' vs '.join(metadata['teams'])}",
                "label_author": "DFL / Sportec Solutions; distributed by Bassek et al. (2025)",
                "label_method": "Native provider BallStatus=0 -> D; status=1 with BallPossession=1/2 -> H/A; otherwise X. No inferred/event-filled labels. " + PAPER,
                "labelled_frames": int(np.count_nonzero(labels != "X")),
                "competition_match_excluded": True, "semantics_compatible": True,
                "blocks_origin": "period_metadata", "input_origin": "observed",
                "target_labels_used_in_inputs": False, "label_origin": "corrected_tracking_derivation",
                "adaptation": "Native 25 Hz; concatenate FrameSet halves, no halftime frames. Metres centred on pitch -> normalized x/L+.5,y/W+.5. S km/h -> m/s. Ignore ball height and all reference fields for features; no tactical tags.",
                "limitations": "Provider control/stoppage reference, not re-annotated under challenge guidelines. Multicamera tracking domain differs from challenge ATD. Pilot split is not sufficient for promotion.",
                "source_files": [{"url": f"https://ndownloader.figshare.com/files/{info_id}", "sha256": sha256(info)},
                                 {"url": f"https://ndownloader.figshare.com/files/{position_id}", "sha256": sha256(positions)}],
                "adapter_sha256": adapter_hash,
            },
        })
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--development", nargs="+", default=["J03WMX"])
    parser.add_argument("--final-test", nargs="+", default=["J03WN1"])
    args = parser.parse_args()
    path = import_dataset(args.source_dir, args.outdir, args.development, args.final_test)
    print(f"IDSSE manifest written: {path}. Final-test scores were not evaluated.")


if __name__ == "__main__":
    main()
