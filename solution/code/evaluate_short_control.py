"""One fixed development test for contact-supported transition costs (experiment 007)."""

import argparse
import hashlib
import json
from pathlib import Path

from . import independent_evaluation as evaluation

ROOT = Path(__file__).resolve().parents[2]


def run(manifest, output):
    manifest, output = Path(manifest), Path(output)
    target = output / "results.json"
    if target.exists():
        raise FileExistsError("Preserve reports; use a new output directory")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if {m["match_id"] for m in payload["matches"]} != {"DFL-MAT-J03WMX", "DFL-MAT-J03WOH"} or any(
            m["split"] != "development" for m in payload["matches"]):
        raise ValueError("Only the declared development matches may be evaluated")
    protocol_path = ROOT / "solution/experiments/short-control-007.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    profiles = ("improved", "short_control")
    report = evaluation.evaluate_manifest(manifest, purpose="development", profiles=profiles, baseline="improved",
                                         diagnostics=True, timing_runs=3)
    previous = json.loads((ROOT / "solution/experiments/possession-improvements-006.results.json").read_text(encoding="utf-8"))
    hashes = {m["match_id"]: m["variants"]["improved"]["prediction_sha256"] for m in previous["clean"]["matches"]}
    for row in report["matches"]:
        if row["prediction_labels_ascii_sha256"]["improved"] != hashes[row["match_id"]]:
            raise AssertionError("The improved baseline changed since experiment 006")
    report["baseline_matches_experiment_006"] = True
    report["ball_gap_stress"] = evaluation.evaluate_manifest(manifest, purpose="development", profiles=profiles,
                                baseline="improved", diagnostics=True, stress_seed=606, stress_mode="ball_gaps")
    report["protocol"] = protocol
    report["protocol_sha256"] = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    report["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    means = report["strata"]["reference"]["equal_match_mean"]
    def pooled(profile, key):
        return sum(row["metrics"][profile]["spell_f1"][0.5][key] for row in report["matches"])
    short = {profile: sum(row["diagnostics"][profile]["duration_bins_seconds"]["0_2s"]["matched_iou_05_count"]
                         for row in report["matches"]) for profile in profiles}
    limits = protocol["promotion_gates"]
    report["gates"] = {
        "short_matches_gain": short["short_control"] - short["improved"] >= limits["minimum_additional_short_matches"],
        "spell_mean_gain": means["short_control"]["spell_f1_mean"] - means["improved"]["spell_f1_mean"] >= limits["minimum_mean_spell_f1_gain"],
        "no_per_match_spell_loss": all(row["paired_delta"]["short_control"]["spell_f1_mean"] >= 0 for row in report["matches"]),
        "macro_guard": all(row["paired_delta"]["short_control"]["macro_f1"] >= -limits["maximum_per_match_macro_f1_loss"] for row in report["matches"]),
        "dead_guard": all(row["metrics"]["short_control"]["per_class"]["D"]["recall"] - row["metrics"]["improved"]["per_class"]["D"]["recall"] >= -limits["maximum_per_match_dead_recall_loss"] for row in report["matches"]),
        "false_positive_guard": pooled("short_control", "n_spurious") - pooled("improved", "n_spurious") <= limits["maximum_pooled_extra_false_positives"],
    }
    report["accepted_on_development"] = all(report["gates"].values())
    output.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    for profile in profiles:
        print(profile, "spellF1", means[profile]["spell_f1_mean"], "short matches", short[profile],
              "TP/FP/FN", pooled(profile,"n_matched"), pooled(profile,"n_spurious"), pooled(profile,"n_missed"))
    print("Development gates:", report["gates"], "accepted:", report["accepted_on_development"])
    print("Full report:", target)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "solution/reports/improvements-006/position-data/manifest.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.manifest, args.output_dir)


if __name__ == "__main__":
    main()
