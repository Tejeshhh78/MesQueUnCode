# MesQueUnCode – Possession from Noisy Tracking

A local solution for **More than a Hack 2027, Phase 1**. It combines geometric
control evidence, automatic tactical hints, unsupervised movement regimes and a
temporal state decoder. No human training labels, model downloads, external
services or GPU are required.

**Verified:** a complete CSV with 199,319 rows, official format validation,
16 synthetic tests and byte-identical regeneration. Initial public pipeline
check: spellF1 **0.4444**, macroF1 **0.7362**. The public window contains only
17 possession spells; private full-match quality is unknown.
See the [technical report](TECHNICAL_REPORT.md) for details and limitations.
For a step-by-step team explanation of the algorithm, scoring, measured weaknesses
and development options, read the [algorithm guide](ALGORITHM_GUIDE.md).

## Run the solution

Run all commands from the repository root, not from inside `solution/`.

Tested with **CPython 3.12.14**. The original challenge dependencies are unchanged;
the solution has a separate file with pinned versions.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\python.exe -m pip install -r solution/requirements.txt
.venv\Scripts\python.exe -m unittest discover -s solution/tests -v
.venv\Scripts\python.exe -m solution.code.possession --team MesQueUnCode --output-dir solution/results
.venv\Scripts\python.exe evaluation/validate_submission.py solution/results/MesQueUnCode.csv
```

On Linux/macOS, the interpreter is `.venv/bin/python`. Place the tactical ATD XML,
ATD text and Smart Tagging XML under `data/tactical/`, as described in
[DATA.md](../docs/DATA.md). Alternatively, use `--data-dir PATH_TO_BUNDLE_ROOT` or set
`MTAH_DATA_DIR`. Videos, broadcast files and homographies are not required.

The output includes:

- `solution/results/MesQueUnCode.csv`: possession labels to submit.
- `solution/results/MesQueUnCode.md`: a copy of the technical report to submit.
- `solution/results/MesQueUnCode.manifest.json`: input/code checksums, configuration,
  package versions and the CSV checksum for local reproduction checks.

`MesQueUnCode` is the provisional team name taken from the repository. Check it
against the registered name before submission and use `--team REGISTERED_NAME`
if needed. Filenames allow letters, digits, underscores and hyphens.

## Reproduction and public pipeline check

For a direct comparison of the fixed original `baseline_smooth50` and our current
algorithm on the same match, run:

```powershell
.\.venv\Scripts\python.exe -m solution.code.compare_algorithms --runs 3 --bootstrap 1000
```

This regenerates both CSVs, measures full-match runtime, and evaluates exactly
those two outputs with the unchanged official scorer on the public five-minute
window. The new method additionally uses automatic Smart Tagging. A comparison
table, bootstrap intervals and scope limitations are saved to
`solution/reports/head-to-head/comparison.md` and `comparison.json`; all outputs stay local.
The comparison performs no training, parameter tuning or method selection.

For **possession detection quality**, the same run also creates
`solution/reports/head-to-head/possession_quality.md` and `possession_quality.json`.
These show correctly matched phases, missed phases, unmatched predictions,
precision, recall and F1 for each official temporal overlap threshold.
To re-score the saved predictions without another runtime benchmark:

```powershell
.\.venv\Scripts\python.exe -m solution.code.compare_algorithms --quality-only --bootstrap 1000
```

This requires a previous complete comparison in the selected `--output-dir`.
It verifies both CSV checksums against the recorded run and invokes the official
scorer again. It does not regenerate predictions from any subsequent code edits;
run the complete comparison to evaluate changed code.

On the fixed public five-minute check, the old/new official spellF1 values are
**0.3256 / 0.4444**. At IoU >= 0.5, both methods match **7 of 17** reference
possessions. Unmatched predictions decrease from **19 to 3**, raising precision
from **26.92% to 70.00%**, while recall remains **41.18%**. F1 at this single
threshold is **0.3256 / 0.5185**, distinct from the averaged ranking metric.
The improvement at this threshold is fewer unmatched predictions, not more
recovered phases. These small-sample results do not establish private full-match
superiority and must not be used to tune the method.

Runtime can be compared independently of prediction quality:

```powershell
.\.venv\Scripts\python.exe -m solution.code.benchmark_runtime --runs 3
```

The test performs one warm-up and three measured runs per method, alternates
their order and checks both generated CSVs. It compares complete calls to our
solution and the original `baseline_smooth50` (`--window 50`), including process
startup and file access. Our solution also validates output and creates a
manifest. The median, minimum, maximum and individual timings are saved under
`solution/reports/runtime-benchmark/timings.json`. Avoid running other compute-intensive
programs during the benchmark. It reads no ground truth and measures no
prediction quality.

After implementation optimisation on 26 October 2026, local medians were
**5.62 s for the baseline** and **13.45 s for our solution** over three measured
runs. Our solution still takes about 2.39 times as long; previously it took
26.74 s, or 5.49 times as long. Machine load varies, so inspect the individual
timings as well. The optimised output remains byte-identical.

```bash
python -m solution.code.possession --team MesQueUnCode --output-dir solution/reports/reproduction
python -c "from pathlib import Path; assert Path('solution/results/MesQueUnCode.csv').read_bytes() == Path('solution/reports/reproduction/MesQueUnCode.csv').read_bytes(); print('Byte-identical')"
```

Optional official scorer invocation for a method that has already been fixed:

```bash
python evaluation/run_evaluation.py --gt public --results solution/results --out solution/reports/public-check --bootstrap 1000 --no-detail
```

The public window is exclusively for checking the pipeline, **not** for training
or selecting parameters/method variants. The solution itself reads no GT.
The recorded method decision and final requirements audit are in
[WORKLOG.md](WORKLOG.md).

## Submission and data protection

According to [SUBMISSION.md](../docs/SUBMISSION.md), submit a CSV, a public repository URL
and a short technical document by **17 November 2026**, using the form supplied
by email. No time of day is specified. A Google account and registration are
required; see [RULES.md](../docs/RULES.md) for team rules. No presentation or deployment
is required for this phase.

`data/`, `solution/results/`, `solution/reports/`, `artifacts/`, `.cache/` and `.venv/` are excluded from Git. Match data and
reconstructable intermediate records must not be published or sent to hosted
LLMs. Submit the CSV only through the designated submission channel. The local
implementation publishes or sends nothing.


## Workspace layout

Run the commands above from the repository root. The shared `games.py`, `data/`,
`starter/` and official `evaluation/` stay at the root. Our implementation lives here:

```text
solution/
  code/
    possession/            inference package
    benchmark_runtime.py   full CLI timing comparison
    compare_algorithms.py  fixed quality and runtime comparison
  tests/                   synthetic correctness tests
  requirements.txt         pinned solution dependencies
  results/                 generated submissions, manifests and report copies (ignored)
  reports/                 generated evaluations and comparisons (ignored)
  TECHNICAL_REPORT.md      submission-oriented method description
  ALGORITHM_GUIDE.md       detailed team explanation
  WORKLOG.md               decisions and verification history
```

The comparison scripts also support direct invocation, such as
`python solution/code/compare_algorithms.py --runs 3 --bootstrap 1000`.
Older local outputs under `artifacts/` remain available. To re-score an older
comparison explicitly, use `--quality-only --output-dir artifacts/head-to-head`.

For standalone official evaluation of our generated submissions:

```powershell
.\.venv\Scripts\python.exe evaluation/run_evaluation.py --gt public --results solution/results --out solution --bootstrap 1000 --no-detail
```

This writes `solution/scoreboard.csv` and `solution/reports/`, separately from the
reference evaluation. CSVs and all generated solution outputs remain local.
