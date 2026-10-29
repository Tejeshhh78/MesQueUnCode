"""Evaluate fixed profiles on explicitly declared OTHER matches, never competition GT.

Usage: python -m solution.code.independent_evaluation MANIFEST --development

Manifest version 1 declares a dataset/protocol and complete match records. Each
record supplies match_id/group_id, split, evidence_type, 25 fps, n_frames, pitch_m,
teams, units, half-open period blocks, provenance, inputs and explicit labels.
Inputs may be ATD files or JSON/NPZ Features exports; inline values are synthetic
test fixtures only. The example in solution/tests/fixtures documents the schema.

Provenance declarations and path/hash guards are engineering safeguards, not
proof of independence or protection against dishonest metadata. Proxy/synthetic
scores are separated from reference scores. This program never promotes a model.
"""

import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from evaluation import possession_metrics as pm
from games import GAME, asset_paths
from .possession.data import Features, read_events, read_tracking
from .possession.model import Config, PROFILES, predict, profile_config


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_TYPES = {"reference", "proxy", "synthetic"}
SPLITS = {"training", "development", "final_test"}
UNITS = {"positions": "normalised_pitch", "distances": "metres", "speeds": "metres_per_second"}
DEFAULT_PROFILES = ("incumbent", "reliable")
METRICS = ("spell_f1_mean", "macro_f1", "accuracy", "alive_dead_f1", "team_acc_gt_alive",
           "team_acc_both_alive", "fragmentation", "mean_spell_duration_s",
           "n_gt_spells", "n_pred_spells", "n_graded", "n_abstain_on_graded")


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _identity(value, name, *, file_safe=False):
    value = _text(value, name)
    pattern = r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}" if file_safe else r"[A-Za-z0-9][A-Za-z0-9._:-]{0,119}"
    if not re.fullmatch(pattern, value):
        raise ValueError(f"{name} must be a simple canonical identifier")
    if re.sub(r"[^a-z0-9]", "", value.casefold()).startswith("barath"):
        raise ValueError("Competition match identifiers are excluded from independent evaluation")
    return value


def _protected_roots():
    roots = [ROOT / "evaluation" / "gt", ROOT / "solution" / "results", ROOT / "artifacts",
             ROOT / "solution" / "reports" / "head-to-head"]
    roots += [Path(asset_paths(source=source)["atd_raw"]).parent for source in ("tactical", "broadcast")]
    return [path.resolve() for path in roots]


def _safe_path(value, base, *, must_exist=True):
    path = (base / _text(value, "File path")).resolve()
    if any(path == root or root in path.parents for root in _protected_roots()):
        raise ValueError("Independent inputs cannot come from competition assets, GT or output directories")
    name = re.sub(r"[^a-z0-9]", "", path.name.casefold())
    if name.startswith("barath") or name == "possessionbyframepkl":
        raise ValueError("Known competition filenames are excluded from independent evaluation")
    if must_exist and not path.is_file():
        raise ValueError("A declared independent input file is missing")
    return path


def _protected_git_blobs():
    """Read object IDs only, without reading public/private reference label files."""
    try:
        result = subprocess.run(["git", "ls-tree", "-r", "HEAD", "--", "evaluation/gt"],
                                cwd=ROOT, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return set()
    return {line.split()[2] for line in result.stdout.splitlines() if len(line.split()) >= 3}


def _known_competition_hashes():
    """Read safe checksum metadata only, not the protected raw assets or labels."""
    hashes = set()
    for path in (ROOT / "solution" / "results" / "MesQueUnCode.manifest.json",
                 ROOT / "artifacts" / "MesQueUnCode.manifest.json"):
        if not path.is_file():
            continue
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(metadata, dict) and metadata.get("game") == GAME:
            candidates = metadata.get("inputs_sha256", {})
            if isinstance(candidates, dict):
                hashes.update(value for value in candidates.values()
                              if isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value))
            value = metadata.get("submission_sha256")
            if isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value):
                hashes.add(value)
    return hashes


def _file_record(path, protected_blobs, protected_hashes):
    before = path.stat()
    digest = hashlib.sha256()
    git_digest = hashlib.sha1(f"blob {before.st_size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            git_digest.update(block)
    # Git normalises text line endings. Check CRLF copies of a tracked reference
    # too, but only for small CSVs, without reading the protected original.
    normalised_blob = None
    if path.suffix.casefold() == ".csv" and before.st_size <= 50 * 1024 * 1024:
        content = path.read_bytes().replace(b"\r\n", b"\n")
        normalised_blob = hashlib.sha1(f"blob {len(content)}\0".encode("ascii") + content).hexdigest()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("An independent input changed while it was being fingerprinted")
    if digest.hexdigest() in protected_hashes or git_digest.hexdigest() in protected_blobs or normalised_blob in protected_blobs:
        raise ValueError("Known competition content cannot be used as independent observations or labels")
    return {"sha256": digest.hexdigest(), "bytes": before.st_size}


def _match_contract(match):
    if not isinstance(match, dict):
        raise ValueError("Every match record must be an object")
    _identity(match.get("match_id"), "match_id")
    _identity(match.get("group_id"), "group_id")
    if match.get("split") not in SPLITS or match.get("evidence_type") not in EVIDENCE_TYPES:
        raise ValueError("Each match needs an explicit split and evidence_type")
    if isinstance(match.get("fps"), bool) or match.get("fps") != 25:
        raise ValueError("This harness requires exactly 25 fps; other rates need a separately verified adapter")
    n = _integer(match.get("n_frames"), "n_frames", 1)
    pitch = np.asarray(match.get("pitch_m"), dtype=float)
    if pitch.shape != (2,) or not np.isfinite(pitch).all() or np.any(pitch <= 0):
        raise ValueError("pitch_m must contain two positive finite metre dimensions")
    teams = match.get("teams")
    if not isinstance(teams, list) or len(teams) != 2 or any(not isinstance(t, str) or not t.strip() for t in teams) or teams[0] == teams[1]:
        raise ValueError("teams must list distinct home and away names in metadata order")
    if match.get("units") != UNITS:
        raise ValueError("Explicit normalised-pitch/metre/metres-per-second units are required")
    blocks, periods, previous = [], set(), 0
    if not isinstance(match.get("blocks"), list):
        raise ValueError("Period blocks must be a list")
    for block in match["blocks"]:
        if not isinstance(block, dict):
            raise ValueError("Every period block must be an object")
        period = _integer(block.get("period"), "period", 1)
        start = _integer(block.get("start"), "block start")
        stop = _integer(block.get("stop"), "block stop", 1)
        if period in periods or start < previous or not start < stop <= n:
            raise ValueError("Period blocks must be unique, ordered, disjoint and half-open within n_frames")
        blocks.append((period, start, stop - 1))
        periods.add(period)
        previous = stop
    if not blocks:
        raise ValueError("Explicit period blocks are required")
    provenance = match.get("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("Provenance must be an object")
    for key in ("source", "license", "organiser_access", "match_identity", "label_author", "label_method"):
        _text(provenance.get(key), f"provenance.{key}")
    if re.search(r"(?i)(?<![a-z0-9])bar[-_ ]ath(?![a-z0-9])", provenance["match_identity"]):
        raise ValueError("Competition match identity cannot be relabelled as an external match")
    if provenance.get("competition_match_excluded") is not True:
        raise ValueError("Explicit confirmation that this is not the competition match is required")
    if provenance.get("semantics_compatible") is not True:
        raise ValueError("Controlled-possession and dead-ball label semantics must be explicitly compatible")
    _integer(provenance.get("labelled_frames"), "labelled_frames", 1)
    if provenance.get("blocks_origin") not in {"period_metadata", "synthetic_periods"}:
        raise ValueError("Blocks must come from period metadata, not possession boundaries")
    if provenance.get("input_origin") not in {"observed", "automatic", "synthetic", "target_derived"}:
        raise ValueError("Input origin must be declared")
    if not isinstance(provenance.get("target_labels_used_in_inputs"), bool):
        raise ValueError("Declare whether target labels were used to construct inputs")
    origin = provenance.get("label_origin")
    if match["evidence_type"] == "reference":
        if origin not in {"human_frame_labels", "corrected_tracking_derivation"}:
            raise ValueError("Reference evidence requires explicit frame-label provenance")
        if provenance["target_labels_used_in_inputs"] or provenance["input_origin"] not in {"observed", "automatic"} or provenance["blocks_origin"] != "period_metadata":
            raise ValueError("Reference labels cannot construct observations or period boundaries")
    elif match["evidence_type"] == "synthetic" and origin != "synthetic":
        raise ValueError("Synthetic evidence must declare synthetic labels")
    elif match["evidence_type"] == "proxy" and origin not in {"automatic_proxy", "human_sparse_events"}:
        raise ValueError("Proxy evidence must identify its proxy-label origin")
    return blocks


def _feature_contract(values, match):
    if not isinstance(values, dict):
        raise ValueError("Feature values must be an object")
    n = match["n_frames"]
    shapes = {"ball": (n, 3), "distance": (n, 2), "player_speed": (n, 2),
              "mean_speed": (n,), "fast_fraction": (n,), "player_count": (n,)}
    arrays = {}

    def numeric_tree(value):
        if isinstance(value, list):
            return all(numeric_tree(item) for item in value)
        return value is None or (isinstance(value, (int, float)) and not isinstance(value, bool))

    for name, shape in shapes.items():
        supplied = values.get(name)
        if isinstance(supplied, np.ndarray):
            if supplied.dtype.kind not in "fiu":
                raise ValueError(f"Feature {name} must use real numeric arrays")
        elif not numeric_tree(supplied):
            raise ValueError(f"Feature {name} must contain numbers or null missing values")
        array = np.asarray(supplied, dtype=float)
        if array.shape != shape or np.isinf(array).any():
            raise ValueError(f"Feature {name} has invalid shape or infinite values")
        arrays[name] = array
    count = arrays["player_count"]
    if not np.isfinite(count).all() or np.any(count < 0) or np.any(count > 32767) or np.any(count != np.floor(count)):
        raise ValueError("Visible player counts must be finite non-negative integers")
    if np.any(arrays["distance"] < 0) or np.any(arrays["fast_fraction"] < 0) or np.any(arrays["fast_fraction"] > 1):
        raise ValueError("Distances must be non-negative and fast fractions must lie in [0, 1]")
    arrays["player_count"] = count.astype(np.int16)
    return Features(**arrays, teams=tuple(match["teams"]))


def _input_paths(match, base, *, must_exist=True):
    inputs = match.get("inputs", {})
    if not isinstance(inputs, dict) or not isinstance(match.get("labels"), dict):
        raise ValueError("Inputs and labels must be explicit objects")
    kind = inputs.get("format")
    if kind == "atd":
        paths = {key: _safe_path(inputs.get(key), base, must_exist=must_exist) for key in ("metadata", "tracking")}
    elif kind in {"features_json", "features_npz"}:
        paths = {"features": _safe_path(inputs.get("path"), base, must_exist=must_exist)}
    elif kind == "inline_features" and match["evidence_type"] == "synthetic":
        paths = {}
    else:
        raise ValueError("Inputs must be ATD, features_json/features_npz, or synthetic-only inline_features")
    if inputs.get("tagging") is not None:
        paths["tagging"] = _safe_path(inputs["tagging"], base, must_exist=must_exist)
    labels = match.get("labels", {})
    if "path" in labels and "values" not in labels:
        paths["labels"] = _safe_path(labels["path"], base, must_exist=must_exist)
    elif "values" not in labels or match["evidence_type"] != "synthetic":
        raise ValueError("Explicit external label CSVs are required; inline labels are synthetic-only")
    return paths


def _load_inputs(match, paths):
    if match["inputs"]["format"] == "atd":
        features = read_tracking(paths["metadata"], paths["tracking"], match["n_frames"], match["pitch_m"])
        if tuple(features.teams) != tuple(match["teams"]):
            raise ValueError("Declared home/away team order disagrees with ATD metadata")
    else:
        if match["inputs"]["format"] == "features_npz":
            with np.load(paths["features"], allow_pickle=False) as archive:
                values = {name: archive[name] for name in ("ball", "distance", "player_speed",
                          "mean_speed", "fast_fraction", "player_count")}
        else:
            values = (json.loads(paths["features"].read_text(encoding="utf-8"))
                      if "features" in paths else match["inputs"].get("values", {}))
        features = _feature_contract(values, match)
    events = read_events(paths["tagging"], match["n_frames"], 25) if "tagging" in paths else []
    return features, events


def _read_labels(match, paths, blocks):
    if "labels" in paths:
        try:
            table = pm.read_label_csv(paths["labels"])
            if list(table.columns) != ["frame", "label"] or pm.validate_submission(table, match["n_frames"]):
                raise ValueError("Invalid label CSV")
            codes = pm.label_codes(table)
        except (OSError, ValueError, TypeError):
            raise ValueError("Independent label CSV violates the explicit frame/label contract") from None
    else:
        values = match["labels"]["values"]
        if not isinstance(values, list) or len(values) != match["n_frames"] or any(v not in pm.LABELS for v in values):
            raise ValueError("Synthetic labels must explicitly supply one H/A/D/X value per frame")
        codes = pm.label_codes(values)
    observed = np.zeros(len(codes), dtype=bool)
    for _, start, last in blocks:
        observed[start:last + 1] = True
    if np.any(codes[~observed] != pm.X):
        raise ValueError("Reference labels outside declared period blocks must be X")
    if int(np.sum(codes != pm.X)) != match["provenance"]["labelled_frames"]:
        raise ValueError("Declared labelled_frames does not match the explicit labels")
    return codes


def _aggregate_metrics(result):
    output = {key: result[key] for key in METRICS}
    output.update({key: result[key] for key in ("per_class", "spell_f1", "spell_f1_at_headline",
                    "spell_bootstrap_low", "spell_bootstrap_high", "bootstrap_low", "bootstrap_high")})
    output["transitions"] = {str(tau): {key: value for key, value in values.items()
                              if key not in {"pairs", "missed", "spurious"}}
                             for tau, values in result["transitions"].items()}
    # The scorer uses a numeric sentinel when no alive frames are predicted.
    # Expose its definedness; do not average that sentinel as a real share error.
    output["share_defined"] = result["share_defined"]
    output["share_err"] = result["share_err"] if result["share_defined"] else None
    return output


def _score_grid(reference, predictions, blocks):
    """Separate metadata periods for the unchanged official temporal scorer.

    Every original frame, including unknown-reference gaps, is retained. Only
    ignored X padding is inserted immediately before each later period. Inputs,
    inference, period-internal durations and paired transition offsets do not
    change. Padding exceeds twice the largest official transition tolerance so
    global transition matching cannot connect different metadata periods.
    """
    reference = np.asarray(reference)
    if reference.ndim != 1 or any(np.asarray(values).shape != reference.shape for values in predictions.values()):
        raise ValueError("Reference and predictions must share a one-dimensional source timeline")
    padding = 2 * max(pm.TAUS) + 1
    cuts = [start for _, start, _ in blocks[1:]]
    pieces = []
    cursor = 0
    for cut in cuts:
        pieces.extend((slice(cursor, cut), None))
        cursor = cut
    pieces.append(slice(cursor, len(reference)))

    def expand(values):
        return np.concatenate([np.full(padding, pm.X, dtype=values.dtype)
                               if span is None else values[span] for span in pieces])

    scored_reference = expand(reference)
    scored_predictions = {profile: expand(np.asarray(values)) for profile, values in predictions.items()}
    return scored_reference, scored_predictions, {
        "source_fps": 25, "score_grid_fps": 25, "source_frames": len(reference),
        "score_grid_frames": len(scored_reference), "metadata_periods": len(blocks),
        "padding_per_boundary_frames": padding, "inserted_ignored_frames": padding * len(cuts),
        "policy": "metadata-period X padding only; every source frame and within-period unknown gap retained",
    }


def _manifest_payload(manifest_path):
    path = Path(manifest_path).resolve()
    if any(root == path or root in path.parents for root in _protected_roots()):
        raise ValueError("Independent manifests cannot come from competition directories")
    payload = path.read_bytes()
    manifest = json.loads(payload)
    if not isinstance(manifest, dict):
        raise ValueError("The independent manifest must be an object")
    return path, payload, manifest


def evaluate_manifest(manifest_path, *, purpose, profiles=DEFAULT_PROFILES, configs=None, bootstrap_n=0):
    """Return aggregate results only; labels are parsed after all profile predictions."""
    if purpose not in {"development", "final_test"}:
        raise ValueError("Choose development or final_test explicitly")
    if not profiles or len(set(profiles)) != len(profiles) or "incumbent" not in profiles or any(p not in PROFILES for p in profiles):
        raise ValueError("Profiles must be unique, include incumbent, and use supported model profiles")
    _integer(bootstrap_n, "bootstrap_n")
    path, manifest_bytes, manifest = _manifest_payload(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError("Expected independent manifest schema_version 1")
    dataset_id = _identity(manifest.get("dataset_id"), "dataset_id", file_safe=True)
    protocol = manifest.get("protocol", {})
    if not isinstance(protocol, dict):
        raise ValueError("The protocol must be an object")
    _text(protocol.get("id"), "protocol.id")
    _text(protocol.get("selection_rule"), "protocol.selection_rule")
    if protocol.get("declared_before_results") is not True or protocol.get("split_unit") != "match":
        raise ValueError("Declare a pre-result protocol with whole-match splits")
    if purpose == "final_test" and protocol.get("final_test_locked") is not True:
        raise ValueError("Final-test evaluation requires an explicitly locked final-test protocol")
    records, identities, groups, input_paths = [], set(), set(), set()
    matches = manifest.get("matches", [])
    if not isinstance(matches, list) or not matches:
        raise ValueError("The manifest must contain match records")
    for match in matches:
        blocks = _match_contract(match)
        # Do not open or require converted inputs/labels for an unselected split.
        paths = _input_paths(match, path.parent, must_exist=match["split"] == purpose)
        identity, group = (match[key].casefold() for key in ("match_id", "group_id"))
        if identity in identities or group in groups:
            raise ValueError("Repeated physical matches/views cannot appear as independent match records")
        source_path = paths.get("tracking", paths.get("features"))
        if source_path is not None and source_path in input_paths:
            raise ValueError("The same observations cannot be reused across independent match records")
        if source_path is not None:
            input_paths.add(source_path)
        identities.add(identity)
        groups.add(group)
        if match["split"] == purpose:
            if purpose == "final_test" and match["evidence_type"] != "reference":
                raise ValueError("Final tests require independent reference evidence, not proxy/synthetic labels")
            records.append((match, blocks, paths))
    if not records:
        raise ValueError("No matches belong to the explicitly selected split")
    protected_blobs, protected_hashes = _protected_git_blobs(), _known_competition_hashes()
    sources = [Path(__file__), *sorted((ROOT / "solution/code/possession").glob("*.py")),
               ROOT / "evaluation/possession_metrics.py", ROOT / "games.py", ROOT / "solution/requirements.txt"]
    source_hashes = {file.relative_to(ROOT).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest() for file in sources}
    configurations = {profile: (configs or {}).get(profile, profile_config(profile)) for profile in profiles}
    rows = []
    for match, blocks, paths in records:
        fingerprints = {role: _file_record(file, protected_blobs, protected_hashes) for role, file in paths.items()}
        features, events = _load_inputs(match, paths)
        predictions, seconds = {}, {}
        for profile in profiles:
            inputs = copy.deepcopy(features)
            tags = copy.deepcopy(events)
            kwargs = {"config": configurations[profile]}
            if "profile" in inspect.signature(predict).parameters:
                kwargs["profile"] = profile
            elif profile != "incumbent":
                raise ValueError("This model revision does not implement experimental profile selection")
            started = time.perf_counter()
            labels, _ = predict(inputs, tags, blocks, match["pitch_m"], 25, **kwargs)
            seconds[profile] = time.perf_counter() - started
            labels = np.asarray(labels)
            if labels.shape != (match["n_frames"],) or not np.isin(labels, pm.LABELS).all():
                raise ValueError("A profile returned invalid labels or frame coverage")
            predictions[profile] = labels.copy()
        # Target labels are not an argument to input construction or inference.
        reference = _read_labels(match, paths, blocks)
        score_reference, score_predictions, timebase = _score_grid(
            reference, {profile: pm.label_codes(labels) for profile, labels in predictions.items()}, blocks)
        scores = {profile: _aggregate_metrics(pm.evaluate(score_reference, labels, bootstrap_n=bootstrap_n))
                  for profile, labels in score_predictions.items()}
        if fingerprints != {role: _file_record(file, protected_blobs, protected_hashes) for role, file in paths.items()}:
            raise ValueError("Independent files changed during evaluation")
        rows.append({"match_id": match["match_id"], "group_id": match["group_id"],
                     "evidence_type": match["evidence_type"], "provenance": match["provenance"],
                     "input_sha256": fingerprints, "n_frames": match["n_frames"], "fps": 25,
                     "evaluation_timebase": timebase,
                     "metrics": scores, "predict_seconds": seconds,
                     "prediction_labels_ascii_sha256": {profile: hashlib.sha256("".join(labels.tolist()).encode("ascii")).hexdigest()
                                                        for profile, labels in predictions.items()},
                     "paired_delta": {profile: {metric: scores[profile][metric] - scores["incumbent"][metric]
                                                  for metric in METRICS}
                                      for profile in profiles if profile != "incumbent"}})
    strata = {}
    for evidence in sorted({row["evidence_type"] for row in rows}):
        subset = [row for row in rows if row["evidence_type"] == evidence]
        strata[evidence] = {"matches": len(subset), "equal_match_mean": {
            profile: {metric: float(np.mean([row["metrics"][profile][metric] for row in subset])) for metric in METRICS}
            for profile in profiles}, "paired_mean_delta": {
            profile: {metric: float(np.mean([row["paired_delta"][profile][metric] for row in subset])) for metric in METRICS}
            for profile in profiles if profile != "incumbent"}}
    if path.read_bytes() != manifest_bytes:
        raise ValueError("Manifest changed during evaluation")
    if source_hashes != {file.relative_to(ROOT).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest() for file in sources}:
        raise ValueError("Inference or evaluation source changed during the run")
    return {"schema_version": 1, "dataset_id": dataset_id, "purpose": purpose, "protocol": protocol,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "source_sha256": source_hashes,
            "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
            "profiles": {profile: configurations[profile].to_dict() for profile in profiles},
            "runtime_scope": "predict_only_excludes_loading_and_input_copying",
            "automatic_promotion": False, "provenance_is_self_declared": True,
            "guard_coverage": {"protected_git_blobs": len(protected_blobs), "known_competition_hashes": len(protected_hashes)},
            "limitations": ["Guards are conservative engineering policies, not proof of independence or anti-fraud security.",
                            "Proxy and synthetic scores do not establish real-match accuracy.",
                            "Separate per-match bootstrap intervals do not prove paired or cross-match significance."],
            "strata": strata, "matches": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--development", action="store_true")
    mode.add_argument("--final-test", action="store_true")
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(DEFAULT_PROFILES))
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "solution/reports/independent")
    args = parser.parse_args(argv)
    purpose = "development" if args.development else "final_test"
    try:
        _, _, manifest = _manifest_payload(args.manifest)
        dataset_id = _identity(manifest.get("dataset_id"), "dataset_id", file_safe=True)
        target = args.output_dir / f"{dataset_id}-{purpose}.json"
        if target.exists():
            raise ValueError("Report already exists; preserve it and use a different output directory")
        report = evaluate_manifest(args.manifest, purpose=purpose, profiles=tuple(args.profiles), bootstrap_n=args.bootstrap)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        # Preserve previous experiments and final-test records rather than silently overwriting.
        with target.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        print(f"Evaluated {len(report['matches'])} declared other-match records for {purpose}.")
        for evidence, values in report["strata"].items():
            print(f"Evidence type: {evidence}; matches: {values['matches']}")
            for profile, metrics in values["equal_match_mean"].items():
                print(f"  {profile}: mean spellF1 {metrics['spell_f1_mean']:.4f}; macroF1 {metrics['macro_f1']:.4f}")
        print("No automatic promotion or real-match superiority claim is made.")
        print(f"Aggregate report: {target}")
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(1, f"ERROR: {error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
