"""Fixed development comparison for experiment 006; never opens final-test games.

python -m solution.code.evaluate_improvements --output-dir solution/reports/improvements-006
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

from . import independent_evaluation as evaluation
from .import_idsse import import_dataset
from .possession.model import PROFILES

ROOT = Path(__file__).resolve().parents[2]
BASELINE_COMMIT = "fcacb57ac25cb53344bb4ce6114954b4550b26bd"


def original_model():
    """Load the reviewed baseline from its immutable local Git object."""
    source = subprocess.run(["git", "show", f"{BASELINE_COMMIT}:solution/code/possession/model.py"],
                            cwd=ROOT, check=True, capture_output=True).stdout
    module = types.ModuleType("solution.code.possession._experiment006_baseline")
    module.__package__ = "solution.code.possession"
    sys.modules[module.__name__] = module
    exec(compile(source, f"{BASELINE_COMMIT}:model.py", "exec"), module.__dict__)
    return module, hashlib.sha256(source).hexdigest()


def run(output, source_dir):
    output = Path(output)
    target = output / "results.json"
    if target.exists():
        raise FileExistsError("Preserve existing results; choose a fresh output directory")
    manifest = output / "position-data" / "manifest.json"
    if not manifest.exists():
        manifest = import_dataset(source_dir, manifest.parent, ["J03WMX", "J03WOH"], [], position_motion=True)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if {m["match_id"] for m in payload["matches"]} != {"DFL-MAT-J03WMX", "DFL-MAT-J03WOH"}:
        raise ValueError("Experiment 006 permits only the two declared development games")
    if any(m["split"] != "development" for m in payload["matches"]):
        raise ValueError("This experiment never evaluates final games")
    protocol_path = ROOT / "solution/experiments/possession-improvements-006.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    profiles = tuple(protocol["variants"])
    report = evaluation.evaluate_manifest(manifest, purpose="development", profiles=profiles,
                                         baseline="controlled", diagnostics=True, bootstrap_n=0, timing_runs=3)
    old, old_hash = original_model()
    for row, match in zip(report["matches"], payload["matches"]):
        paths = evaluation._input_paths(match, manifest.parent)
        features, events = evaluation._load_inputs(match, paths)
        blocks = evaluation._match_contract(match)
        labels, _ = old.predict(features, events, blocks, match["pitch_m"], 25)
        actual = hashlib.sha256("".join(labels.tolist()).encode("ascii")).hexdigest()
        if actual != row["prediction_labels_ascii_sha256"]["controlled"]:
            raise AssertionError("Controlled no longer reproduces the original decoder on identical inputs")
        row["original_model_predictions_identical"] = True
    report["experiment"] = protocol
    report["protocol_sha256"] = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    report["original_model_source_sha256"] = old_hash
    report["baseline_commit"] = BASELINE_COMMIT
    report["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    stress = evaluation.evaluate_manifest(manifest, purpose="development", profiles=profiles,
                                         baseline="controlled", diagnostics=True, bootstrap_n=0,
                                         stress_seed=606, stress_mode="ball_gaps")
    report["ball_gap_stress"] = stress
    means = report["strata"]["reference"]["equal_match_mean"]
    gates = {}
    for profile in profiles[1:]:
        deltas = [row["paired_delta"][profile] for row in report["matches"]]
        gates[profile] = {
            "positive_mean_spell_f1": means[profile]["spell_f1_mean"] > means["controlled"]["spell_f1_mean"],
            "no_per_match_spell_f1_loss": all(d["spell_f1_mean"] >= 0 for d in deltas),
            "macro_f1_guard": all(d["macro_f1"] >= -0.005 for d in deltas),
            "dead_recall_guard": all(row["metrics"][profile]["per_class"]["D"]["recall"]
                                     - row["metrics"]["controlled"]["per_class"]["D"]["recall"] >= -0.02
                                     for row in report["matches"]),
        }
    report["development_gates"] = gates
    # No bootstrap claim from two reused development games; keep scores descriptive.
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print("profile             spellF1  macroF1   development gates")
    for profile, metrics in means.items():
        status = "baseline" if profile == "controlled" else str(all(gates[profile].values()))
        print(f"{profile:19} {metrics['spell_f1_mean']:.6f} {metrics['macro_f1']:.6f}  {status}")
    print(f"Original model predictions identical on both games. Full report: {target}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "solution/reports/external/idsse")
    args = parser.parse_args()
    run(args.output_dir, args.source_dir)


if __name__ == "__main__":
    main()
