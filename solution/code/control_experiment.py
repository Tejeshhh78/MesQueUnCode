"""Execute the predeclared responsive switch-cost experiment and one locked final test.

Use ``select --protocol P --manifest M --output-dir O`` for development, then
``final-test --selection O/selection.json --output-dir F`` only after selection.
All outputs are aggregate local records. No command changes the default model,
uses competition scoring, or claims that a selection gate proves superiority.
"""

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys

from . import independent_evaluation as validation
from .possession.model import profile_config


ROOT = Path(__file__).resolve().parents[2]
POLICY = {
    "candidate_costs": [4.0, 8.0, 12.0], "primary_metric": "equal_match_mean_spell_f1_mean",
    "minimum_development_matches": 2, "minimum_held_out_matches": 2,
    "minimum_primary_gain": 0.01, "maximum_macro_f1_loss": 0.005,
    "maximum_dead_ball_recall_loss": 0.02, "maximum_iou_05_recall_loss": 0.02,
    "maximum_transition_12_f1_loss": 0.02, "maximum_predict_runtime_ratio": 1.5,
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Experiment records must be JSON objects")
    return value


def _write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _empty_output(path):
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError("Experiment output already contains records; preserve it")


def _source_snapshot():
    files = [Path(__file__), Path(validation.__file__), ROOT / "solution/code/import_idsse.py",
             *sorted((ROOT / "solution/code/possession").glob("*.py")),
             ROOT / "evaluation/possession_metrics.py", ROOT / "games.py", ROOT / "solution/requirements.txt"]
    return {file.relative_to(ROOT).as_posix(): _file_hash(file) for file in files}


def _protocol(path):
    record = _read_json(path)
    if record.get("id") != "control-switch-cost-002" or record.get("declared_before_results") is not True:
        raise ValueError("This driver requires the explicitly predeclared control-switch-cost-002 protocol")
    if any(record.get(key) != value for key, value in POLICY.items()):
        raise ValueError("The protocol differs from the predeclared candidate grid or acceptance gates")
    groups = []
    for split in ("development", "final_test"):
        ids = record.get(split)
        if not isinstance(ids, list) or any(not isinstance(item, str) or not item for item in ids) or len(set(ids)) != len(ids):
            raise ValueError("Protocol splits must explicitly list unique match identities")
        groups.append(set(ids))
    if groups[0] & groups[1]:
        raise ValueError("Development and held-out identities must be disjoint")
    if len(groups[0]) < 2 or len(groups[1]) < 2:
        raise ValueError("The predeclared protocol requires at least two development and two held-out matches")
    return record


def _split_contract(manifest, protocol):
    """Inspect declarations only; do not open either split's observations or labels."""
    for split in ("development", "final_test"):
        matches = [row for row in manifest.get("matches", []) if row.get("split") == split]
        ids = [row.get("match_id") for row in matches]
        if len(set(ids)) != len(ids) or set(ids) != set(protocol[split]):
            raise ValueError("Manifest match identities do not equal the predeclared split")
        if any(row.get("evidence_type") != "reference" for row in matches):
            raise ValueError("Selection and final confirmation require declared independent reference evidence")


def _configuration(cost):
    incumbent = profile_config("incumbent")
    if (incumbent.team_switch_cost, incumbent.stoppage_switch_cost, incumbent.min_state_s) != (20.0, 24.0, 0.32):
        raise ValueError("The incumbent no longer matches the predeclared temporal assumptions")
    return {"incumbent": incumbent, "responsive": replace(incumbent, team_switch_cost=float(cost))}


def _values(row, profile):
    metrics = row["metrics"][profile]
    iou = metrics["spell_f1"].get(0.5, metrics["spell_f1"].get("0.5"))
    values = {"primary": metrics["spell_f1_mean"], "macro": metrics["macro_f1"],
              "dead_recall": metrics["per_class"]["D"]["recall"], "iou_recall": iou["recall"],
              "transition": metrics["transitions"]["12"]["f1"], "seconds": row["predict_seconds"][profile]}
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values.values()):
        raise ValueError("Selection requires finite official metrics and measured prediction runtimes")
    if values["seconds"] <= 0:
        raise ValueError("Measured prediction runtimes must be positive")
    return values


def assess_candidate(report, protocol, *, purpose="development"):
    """Apply fixed per-match regression gates and equal-match primary gain."""
    rows = report["matches"]
    expected = set(protocol[purpose])
    if len(rows) != len(expected) or {row["match_id"] for row in rows} != expected or any(row["evidence_type"] != "reference" for row in rows):
        raise ValueError("Evaluation does not cover exactly the predeclared independent reference matches")
    checks, before, after = [], [], []
    for row in rows:
        old, new = _values(row, "incumbent"), _values(row, "responsive")
        differences = {key: new[key] - old[key] for key in old if key != "seconds"}
        ratio = new["seconds"] / old["seconds"]
        gates = {"nonnegative_primary": differences["primary"] >= -1e-12,
                 "macro_limit": differences["macro"] >= -protocol["maximum_macro_f1_loss"] - 1e-12,
                 "dead_recall_limit": differences["dead_recall"] >= -protocol["maximum_dead_ball_recall_loss"] - 1e-12,
                 "iou_recall_limit": differences["iou_recall"] >= -protocol["maximum_iou_05_recall_loss"] - 1e-12,
                 "transition_limit": differences["transition"] >= -protocol["maximum_transition_12_f1_loss"] - 1e-12,
                 "runtime_limit": ratio <= protocol["maximum_predict_runtime_ratio"] + 1e-12}
        checks.append({"match_id": row["match_id"], "delta": differences, "runtime_ratio": ratio, "gates": gates})
        before.append(old["primary"])
        after.append(new["primary"])
    old_mean, new_mean = sum(before) / len(before), sum(after) / len(after)
    gain = new_mean - old_mean
    mean_gate = gain >= protocol["minimum_primary_gain"] - 1e-12
    return {"eligible": mean_gate and all(all(row["gates"].values()) for row in checks),
            "incumbent_equal_match_mean": old_mean, "candidate_equal_match_mean": new_mean,
            "equal_match_primary_gain": gain, "mean_gain_gate": mean_gate, "per_match": checks}


def select_result(assessments):
    """Maximise the fixed primary metric; exact ties prefer the larger switch cost."""
    eligible = [(float(cost), item) for cost, item in assessments.items() if item["eligible"]]
    return max(eligible, key=lambda pair: (pair[1]["candidate_equal_match_mean"], pair[0]))[0] if eligible else None


def _unchanged(protocol_path, manifest_path, protocol_hash, manifest_hash, sources):
    if _file_hash(protocol_path) != protocol_hash or _file_hash(manifest_path) != manifest_hash or _source_snapshot() != sources:
        raise ValueError("Protocol, dataset manifest or executed source changed after freezing")


def run_development(protocol_path, manifest_path, output_dir):
    protocol_path, manifest_path, output = Path(protocol_path).resolve(), Path(manifest_path).resolve(), Path(output_dir).resolve()
    protocol = _protocol(protocol_path)
    _, _, manifest = validation._manifest_payload(manifest_path)
    _split_contract(manifest, protocol)
    _empty_output(output)
    protocol_hash, manifest_hash, sources = _file_hash(protocol_path), _file_hash(manifest_path), _source_snapshot()
    started = {"protocol_sha256": protocol_hash, "manifest_sha256": manifest_hash, "source_sha256": sources}
    # Anchor consumption to the shared dataset location, not a replaceable output
    # directory. These are accidental-reuse guards, not tamper-proof security.
    _write_new(manifest_path.parent / f".{protocol['id']}-development.started.json", started)
    _write_new(output / "development.started.json", started)
    assessments, reference_fingerprint = {}, None
    for cost in protocol["candidate_costs"]:
        _unchanged(protocol_path, manifest_path, protocol_hash, manifest_hash, sources)
        configs = _configuration(cost)
        report = validation.evaluate_manifest(manifest_path, purpose="development", profiles=("incumbent", "responsive"), configs=configs)
        _unchanged(protocol_path, manifest_path, protocol_hash, manifest_hash, sources)
        fingerprint = {row["match_id"]: {"inputs": row["input_sha256"], "incumbent_metrics": row["metrics"]["incumbent"],
                                        "incumbent_prediction": row["prediction_labels_ascii_sha256"]["incumbent"]}
                       for row in report["matches"]}
        if reference_fingerprint is not None and fingerprint != reference_fingerprint:
            raise ValueError("Independent observations or incumbent predictions differ across the fixed grid")
        reference_fingerprint = fingerprint
        assessment = assess_candidate(report, protocol)
        assessments[str(cost)] = assessment
        _write_new(output / f"candidate-{cost:g}.json", {"cost": cost, "assessment": assessment, "evaluation": report})
    selected = select_result(assessments)
    configurations = None if selected is None else {key: value.to_dict() for key, value in _configuration(selected).items()}
    result = {"schema_version": 1, "record_type": "locked_development_selection", "protocol": protocol,
              "protocol_path": str(protocol_path), "protocol_sha256": protocol_hash,
              "manifest_path": str(manifest_path), "manifest_sha256": manifest_hash,
              "source_sha256": sources, "candidate_assessments": assessments,
              "selected_switch_cost": selected, "configurations": configurations,
              "configuration_sha256": None if configurations is None else _digest(configurations),
              "decision": "retain_incumbent" if selected is None else "selected_for_one_held_out_test",
              "automatic_promotion": False,
              "limitations": "Predeclared engineering gates; single-pass runtime is noisy; declarations do not certify dataset independence."}
    result["selection_sha256"] = _digest(result)
    _write_new(output / "selection.json", result)
    return result


def run_final_test(selection_path, output_dir):
    selection_path, output = Path(selection_path).resolve(), Path(output_dir).resolve()
    selection = _read_json(selection_path)
    signature = selection.pop("selection_sha256", None)
    if signature != _digest(selection) or selection.get("record_type") != "locked_development_selection":
        raise ValueError("The locked selection record was changed or is invalid")
    if selection.get("decision") != "selected_for_one_held_out_test" or selection.get("selected_switch_cost") is None:
        raise ValueError("No candidate qualified; keep the incumbent and do not open final-test labels")
    protocol_path, manifest_path = Path(selection["protocol_path"]), Path(selection["manifest_path"])
    protocol = _protocol(protocol_path)
    if selection["selected_switch_cost"] not in protocol["candidate_costs"]:
        raise ValueError("The selected cost is outside the predeclared grid")
    _unchanged(protocol_path, manifest_path, selection["protocol_sha256"], selection["manifest_sha256"], selection["source_sha256"])
    configs = _configuration(selection["selected_switch_cost"])
    serialised = {key: value.to_dict() for key, value in configs.items()}
    if serialised != selection["configurations"] or _digest(serialised) != selection["configuration_sha256"]:
        raise ValueError("The selected configuration changed before final evaluation")
    _empty_output(output)
    # All selections for this dataset share the held-out consumption marker.
    # Moving selection/output files cannot accidentally reopen the final labels.
    _write_new(manifest_path.parent / ".held-out-final-test.started.json", {
        "selection_sha256": signature, "manifest_sha256": selection["manifest_sha256"],
        "held_out_matches": protocol["final_test"]})
    report = validation.evaluate_manifest(manifest_path, purpose="final_test", profiles=("incumbent", "responsive"), configs=configs)
    _unchanged(protocol_path, manifest_path, selection["protocol_sha256"], selection["manifest_sha256"], selection["source_sha256"])
    assessment = assess_candidate(report, protocol, purpose="final_test")
    result = {"selection_sha256": signature, "configuration_sha256": selection["configuration_sha256"],
              "source_sha256": selection["source_sha256"], "manifest_sha256": selection["manifest_sha256"],
              "assessment": assessment, "evaluation": report, "automatic_promotion": False,
              "review_recommendation": "candidate_supported_for_parent_review" if assessment["eligible"] else "retain_incumbent",
              "stopping_rule": "Final test consumed once. No retuning on these results."}
    _write_new(output / "final.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    modes = parser.add_subparsers(dest="mode", required=True)
    select = modes.add_parser("select")
    select.add_argument("--protocol", type=Path, required=True)
    select.add_argument("--manifest", type=Path, required=True)
    select.add_argument("--output-dir", type=Path, required=True)
    final = modes.add_parser("final-test")
    final.add_argument("--selection", type=Path, required=True)
    final.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = (run_development(args.protocol, args.manifest, args.output_dir) if args.mode == "select"
                  else run_final_test(args.selection, args.output_dir))
        if args.mode == "select":
            print(f"Selected switch cost: {result['selected_switch_cost']}; decision: {result['decision']}")
        else:
            print(f"Final-test recommendation: {result['review_recommendation']}")
        print("No automatic default change or statistical superiority claim is made.")
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(1, f"ERROR: {error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
