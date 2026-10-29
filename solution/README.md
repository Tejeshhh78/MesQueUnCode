# MesQueUnCode

Local football-possession inference for More than a Hack 2027, Phase 1.
The system combines tracking geometry, automatic tactical tags and temporal
state decoding. It runs on CPU without an external service.

## Install and run

Run from the repository root. Tested with CPython 3.12.14; on Linux/macOS use
`.venv/bin/python` instead of the Windows interpreter below.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r solution/requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s solution/tests -v
.\.venv\Scripts\python.exe -m solution.code.possession --profile responsive --team-switch-cost 12 --output-dir solution/reports/responsive-12
.\.venv\Scripts\python.exe evaluation/validate_submission.py solution/reports/responsive-12/MesQueUnCode.csv
```

Place the tactical tracking metadata, tracking text and Smart Tagging XML under
`data/tactical/`, following [the official data layout](../docs/DATA.md).
Alternatively set `MTAH_DATA_DIR` or pass `--data-dir BUNDLE_ROOT`.
Videos and broadcast tracking are not required.

| Profile | Behaviour |
|---|---|
| `incumbent` | Original team solution; CLI default; H/A switch cost 20 |
| `responsive` | Independently evaluated candidate; H/A switch cost 12 |
| `reliable` | Experimental isolated ball-position filter; no demonstrated natural-match gain |

The original hackathon `baseline_smooth50` is a separate method under `starter/`.
New development must compare against `responsive` with cost 12.

Outputs are `MesQueUnCode.csv`, `MesQueUnCode.manifest.json` and a technical
Markdown document. The manifest records input/source/output hashes and the
actual configuration. Use `--team REGISTERED_NAME` for the registered team name.
Use separate directories for different configurations; existing outputs with a
different profile or configuration are protected against overwrite.

Without an explicit directory, incumbent outputs go to `solution/results/` and
experimental profiles to `solution/reports/experimental/PROFILE/`.

## Check and compare

Runtime only, without reference labels:

```powershell
.\.venv\Scripts\python.exe -m solution.code.benchmark_runtime --profile responsive --runs 3 --warmup 1 --output-dir solution/reports/runtime-responsive
```

Original baseline versus the already fixed responsive candidate, including the
public pipeline check:

```powershell
.\.venv\Scripts\python.exe -m solution.code.compare_algorithms --profile responsive --runs 3 --warmup 1 --bootstrap 1000 --output-dir solution/reports/head-to-head-responsive
```

The comparison measures full-match runtime but scores quality only on the public
five minutes. It must not select or tune a model. `--quality-only` re-scores saved,
checksum-verified predictions; it requires their recorded `--profile` and does
not rerun inference. For candidate selection use [independent evaluation](INDEPENDENT_VALIDATION.md).

Optional local evidence diagnostics:

```powershell
.\.venv\Scripts\python.exe -m solution.code.possession --profile responsive --diagnostics --output-dir solution/reports/responsive-diagnostics
```

`--save-trace` additionally writes frame arrays and requires `--diagnostics`
and a repository-local, Git-ignored output path. Keep restricted traces local.

## Method, results and development

- [Algorithm guide](ALGORITHM_GUIDE.md): how inference and scoring work.
- [Technical report](TECHNICAL_REPORT.md): method, measured results, sources and limitations.
- [Next development task](QUALITY_IMPROVEMENT_PROMPT.md): staged instructions for improving possession control.

## Submission

The required artifacts are a 199,319-row `frame,label` CSV, a public code
repository and a short technical document. The stated deadline is
17 November 2026; no time of day is specified. Follow the official
[submission instructions](../docs/SUBMISSION.md) and [rules](../docs/RULES.md).

`data/`, `solution/results/`, `solution/reports/`, `artifacts/` and `.cache/`
are ignored. Do not publish restricted match assets or reconstructable traces,
transmit them to hosted LLMs, or use human-produced data about the competition
match for training or selection. The public reference is a pipeline check only.
