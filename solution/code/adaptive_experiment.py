"""Single development experiment 004; deliberately provides no final-test mode."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from . import independent_evaluation as ev


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "solution/experiments/adaptive-transition-004.json"
OUTPUT = ROOT / "solution/reports/adaptive-transition-004"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def assess(report, protocol):
    """Apply predeclared gates without silently promoting on development evidence."""
    baseline, candidate = protocol["baseline"], protocol["candidate"]
    rows = []
    if {r["match_id"] for r in report["matches"]} != set(protocol["development"]):
        raise ValueError("Expected exactly the declared development games")
    for row in report["matches"]:
        old, new = (row["metrics"][p] for p in (baseline,candidate))
        def values(metrics):
            return {"spell_f1": metrics["spell_f1_mean"], "macro_f1": metrics["macro_f1"],
                    "dead_recall": metrics["per_class"]["D"]["recall"],
                    "iou05_recall": metrics["spell_f1"].get(.5,metrics["spell_f1"].get("0.5"))["recall"],
                    "transition12_f1": metrics["transitions"]["12"]["f1"]}
        before, after = values(old), values(new)
        delta = {k: after[k]-before[k] for k in before}
        limits = {"spell_f1": "spell_f1", "macro_f1": "macro_f1", "dead_recall": "dead_recall",
                  "iou05_recall": "iou_05_recall", "transition12_f1": "transition_12_f1"}
        gates = {k: bool(np.isfinite(d) and d >= -protocol[f"maximum_per_match_{limits[k]}_loss"]-1e-12)
                 for k,d in delta.items()}
        times = row["predict_repetitions_seconds"]
        if any(len(times[p]) != 3 or not all(np.isfinite(t) and t > 0 for t in times[p]) for p in (baseline,candidate)):
            raise ValueError("Three finite positive measured times required per profile")
        ratio = float(np.median(times[candidate])/np.median(times[baseline]))
        gates["runtime"] = ratio <= protocol["maximum_median_predict_runtime_ratio"]
        rows.append({"match_id":row["match_id"],"baseline":before,"candidate":after,
                     "delta":delta,"runtime_ratio":ratio,"gates":gates})
    gain = float(np.mean([r["delta"]["spell_f1"] for r in rows]))
    passed = gain >= protocol["minimum_mean_spell_f1_gain"] and all(all(r["gates"].values()) for r in rows)
    return {"development_gates_passed":bool(passed),"mean_spell_f1_gain":gain,
            "mean_gain_gate":gain >= protocol["minimum_mean_spell_f1_gain"],"per_match":rows,
            "promoted":False,"final_evaluated":False,
            "decision":"experimental_pending_new_final_data" if passed else "rejected_on_development"}


def run(manifest_path):
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    path, _, manifest = ev._manifest_payload(manifest_path)
    development = [m for m in manifest["matches"] if m["split"] == "development"]
    if sorted(m["match_id"] for m in development) != sorted(protocol["development"]):
        raise ValueError("Manifest development split differs from protocol")
    if any(m["evidence_type"] != "reference" for m in development):
        raise ValueError("This experiment requires other-match reference data")
    if protocol["id"] != "adaptive-transition-004" or protocol["variant_budget"] != 1:
        raise ValueError("This driver implements only the single declared experiment")
    OUTPUT.mkdir(parents=True,exist_ok=True)
    sources = [*sorted((ROOT/"solution/code").glob("*.py")),
               *sorted((ROOT/"solution/code/possession").glob("*.py")),
               ROOT/"games.py",ROOT/"evaluation/possession_metrics.py",ROOT/"solution/requirements.txt"]
    hashes = {p.relative_to(ROOT).as_posix():digest(p) for p in sources}
    frozen = {"protocol_sha256":digest(PROTOCOL),"manifest_sha256":digest(path),"source_sha256":hashes}
    write_new(OUTPUT/"development-started.json",frozen)
    for p in sources:
        target=OUTPUT/"evaluated-source"/p.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target)
    shutil.copy2(PROTOCOL,OUTPUT/"evaluated-source"/PROTOCOL.name)
    report = ev.evaluate_manifest(path,purpose="development",profiles=("controlled","adaptive"),
                                  baseline="controlled",diagnostics=True,timing_runs=3)
    if frozen != {"protocol_sha256":digest(PROTOCOL),"manifest_sha256":digest(path),
                  "source_sha256":{p.relative_to(ROOT).as_posix():digest(p) for p in sources}}:
        raise ValueError("Protocol, manifest or code changed during experiment")
    baseline=json.loads((OUTPUT/"baseline-diagnosis.json").read_text())
    expected={r["match_id"]:r["prediction_sha256"] for r in baseline["matches"]}
    if any(r["prediction_labels_ascii_sha256"]["controlled"] != expected[r["match_id"]] for r in report["matches"]):
        raise ValueError("Controlled baseline predictions changed")
    report["experiment"] = frozen
    report["assessment"] = assess(report,protocol)
    write_new(OUTPUT/"development.json",report)
    print(json.dumps(report["assessment"],indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest",type=Path)
    args=parser.parse_args()
    try:
        run(args.manifest)
    except (OSError,ValueError,KeyError) as error:
        parser.exit(1,f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
