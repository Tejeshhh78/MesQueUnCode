"""Compare complete CLI runtimes without using ground truth."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import possession_metrics as pm
from games import GAME, GAMES

SOLUTION_ROOT = ROOT / "solution"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3, help="Measured runs per method (default: 3)")
    parser.add_argument("--warmup", type=int, default=1, help="Unmeasured warm-up runs per method")
    parser.add_argument("--data-dir", type=Path, help="Alternative data bundle root for both methods")
    parser.add_argument("--output-dir", type=Path, default=SOLUTION_ROOT / "reports" / "runtime-benchmark")
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 0:
        parser.error("--runs must be positive and --warmup must be at least 0")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    if args.data_dir:
        environment["MTAH_DATA_DIR"] = str(args.data_dir.resolve())
    commands = {
        "baseline_smooth50": [sys.executable, str(ROOT / "starter" / "baseline.py"),
                              "--window", "50", "--out", str(output / "baseline")],
        "possession": [sys.executable, "-m", "solution.code.possession", "--team", "MesQueUnCode",
                       "--output-dir", str(output / "solution")],
    }
    files = {"baseline_smooth50": output / "baseline" / "baseline_smooth50.csv",
             "possession": output / "solution" / "MesQueUnCode.csv"}
    durations = {name: [] for name in commands}
    fingerprints = {}
    print("Complete CLI calls: process startup + reading + computation + output.", flush=True)
    print("The solution also validates output and creates a manifest; runs are sequential.", flush=True)
    for iteration in range(args.warmup + args.runs):
        order = list(commands)
        if iteration % 2:
            order.reverse()
        warmup = iteration < args.warmup
        phase = f"Warm-up {iteration + 1}/{args.warmup}" if warmup else f"Run {iteration - args.warmup + 1}/{args.runs}"
        for name in order:
            started = time.perf_counter()
            result = subprocess.run(commands[name], cwd=ROOT, env=environment,
                                    capture_output=True, text=True, encoding="utf-8", errors="replace")
            elapsed = time.perf_counter() - started
            if result.returncode:
                raise RuntimeError(f"{name} failed: {result.stderr.strip() or result.stdout.strip()}")
            # Apply the same checks to both outputs, outside the timed CLI call.
            reason = pm.validate_submission(pm.read_label_csv(files[name]), GAMES[GAME]["n_video_frames"])
            if reason:
                raise RuntimeError(f"{name}: invalid CSV: {reason}")
            digest = hashlib.sha256(files[name].read_bytes()).hexdigest()
            if name in fingerprints and digest != fingerprints[name]:
                raise RuntimeError(f"{name}: output differs between runs")
            fingerprints[name] = digest
            if not warmup:
                durations[name].append(elapsed)
            print(f"{phase}: {name:20} {elapsed:7.2f} s", flush=True)
    summary = {name: {"seconds": values, "median_s": statistics.median(values),
                       "min_s": min(values), "max_s": max(values), "csv_sha256": fingerprints[name]}
               for name, values in durations.items()}
    factor = summary["possession"]["median_s"] / summary["baseline_smooth50"]["median_s"]
    report = {"python": platform.python_version(), "platform": platform.platform(),
              "runs": args.runs, "warmup": args.warmup, "scope": "complete_cli_wall_time",
              "ground_truth_used": False, "results": summary,
              "solution_time_divided_by_baseline": factor}
    target = output / "timings.json"
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\nMedian:")
    for name, values in summary.items():
        print(f"  {name:20} {values['median_s']:.2f} s  (min {values['min_s']:.2f}, max {values['max_s']:.2f})")
    if factor >= 1:
        print(f"The solution takes {factor:.2f} times as long as the baseline.")
    else:
        print(f"The solution is {1 / factor:.2f} times faster than the baseline.")
    print(f"Timing data: {target}")


if __name__ == "__main__":
    main()
