# MesQueUnCode – Possession from Noisy Tracking

A local solution for **More than a Hack 2027, Phase 1**. It combines geometric
control evidence, automatic tactical hints, unsupervised movement regimes and a
temporal state decoder. No human training labels, model downloads, external
services or GPU are required.

**Verified:** a complete CSV with 199,319 rows, official format validation,
24 synthetic tests and byte-identical regeneration. Initial public pipeline
check: spellF1 **0.4444**, macroF1 **0.7362**. The public window contains only
17 possession spells; private full-match quality is unknown.
See the [technical report](TECHNICAL_REPORT.md) for details and limitations.

## Run the solution

Tested with **CPython 3.12.14**. The original challenge dependencies are unchanged;
the solution has a separate file with pinned versions.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\python.exe -m pip install -r requirements-solution.txt
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m possession --team MesQueUnCode --output-dir artifacts
.venv\Scripts\python.exe evaluation/validate_submission.py artifacts/MesQueUnCode.csv
```

On Linux/macOS, the interpreter is `.venv/bin/python`. Place the tactical ATD XML,
ATD text and Smart Tagging XML under `data/tactical/`, as described in
[DATA.md](DATA.md). Alternatively, use `--data-dir PATH_TO_BUNDLE_ROOT` or set
`MTAH_DATA_DIR`. Videos, broadcast files and homographies are not required.

The output includes:

- `artifacts/MesQueUnCode.csv`: possession labels to submit.
- `artifacts/MesQueUnCode.md`: a copy of the technical report to submit.
- `artifacts/MesQueUnCode.manifest.json`: input/code checksums, configuration,
  package versions and the CSV checksum for local reproduction checks.

`MesQueUnCode` is the provisional team name taken from the repository. Check it
against the registered name before submission and use `--team REGISTERED_NAME`
if needed. Filenames allow letters, digits, underscores and hyphens.

## Reproduction and public pipeline check

For a direct comparison of the fixed original `baseline_smooth50` and our current
algorithm on the same match, run:

```powershell
.\.venv\Scripts\python.exe compare_algorithms.py --runs 3 --bootstrap 1000
```

This regenerates both CSVs, measures full-match runtime, and evaluates exactly
those two outputs with the unchanged official scorer on the public five-minute
window. The new method additionally uses automatic Smart Tagging. A comparison
table, bootstrap intervals and scope limitations are saved to
`artifacts/head-to-head/comparison.md` and `comparison.json`; all outputs stay local.
The comparison performs no training, parameter tuning or method selection.

For **possession detection quality**, the same run also creates
`artifacts/head-to-head/possession_quality.md` and `possession_quality.json`.
These show correctly matched phases, missed phases, unmatched predictions,
precision, recall and F1 for each official temporal overlap threshold.
To re-score the saved predictions without another runtime benchmark:

```powershell
.\.venv\Scripts\python.exe compare_algorithms.py --quality-only --bootstrap 1000
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
.\.venv\Scripts\python.exe benchmark_runtime.py --runs 3
```

The test performs one warm-up and three measured runs per method, alternates
their order and checks both generated CSVs. It compares complete calls to our
solution and the original `baseline_smooth50` (`--window 50`), including process
startup and file access. Our solution also validates output and creates a
manifest. The median, minimum, maximum and individual timings are saved under
`artifacts/runtime-benchmark/timings.json`. Avoid running other compute-intensive
programs during the benchmark. It reads no ground truth and measures no
prediction quality.

After implementation optimisation on 26 October 2026, local medians were
**5.62 s for the baseline** and **13.45 s for our solution** over three measured
runs. Our solution still takes about 2.39 times as long; previously it took
26.74 s, or 5.49 times as long. Machine load varies, so inspect the individual
timings as well. The optimised output remains byte-identical.

```bash
python -m possession --team MesQueUnCode --output-dir artifacts/reproduction
python -c "from pathlib import Path; assert Path('artifacts/MesQueUnCode.csv').read_bytes() == Path('artifacts/reproduction/MesQueUnCode.csv').read_bytes(); print('Byte-identical')"
```

Optional official scorer invocation for a method that has already been fixed:

```bash
python evaluation/run_evaluation.py --gt public --results artifacts --out artifacts/public-check --bootstrap 1000 --no-detail
```

The public window is exclusively for checking the pipeline, **not** for training
or selecting parameters/method variants. The solution itself reads no GT.
The recorded method decision and final requirements audit are in
[WORKLOG.md](WORKLOG.md).

## Validation on other matches (profile v2)

RULES.md allows labelled data from other games. `validation/` scores the model on
the three public [Metrica Sports sample games](https://github.com/metrica-sports/sample-data)
(anonymised matches with tracking and event data, © Metrica Sports). Possession and
dead-ball labels are derived mechanically from each game's own events; the clean
tracking is degraded towards the automatic ATD (ball missing on 35–45 % of live
frames, positional noise, dropouts). It reads nothing of the provided match.

```powershell
.\.venv\Scripts\python.exe -m validation --download     # once; files go to data/external/ (gitignored)
.\.venv\Scripts\python.exe -m validation                # v1, v2 and baseline_smooth50; ~6 min
.\.venv\Scripts\python.exe -m validation --quick        # one seed, one regime, no tags
```

Profile `v2` was selected there: control evidence within ~1.5 m of a player, team
switch cost 15, tag weight 0.6 and switch placement in the middle of ball gaps.
Across 3 games × 2 seeds × 4 tracking-degradation regimes × 3 synthetic tagging
models it beat `v1` in all 72 paired comparisons (spellF1 +0.008 to +0.081, mean
+0.022 to +0.071 per setting). These are other matches with simulated tracking
noise, so the gain on the provided match is not established. `v1` stays the
default and is unchanged; generate `v2` with:

```powershell
.\.venv\Scripts\python.exe -m possession --profile v2 --team MesQueUnCode --output-dir artifacts/v2
```

## Submission and data protection

According to [SUBMISSION.md](SUBMISSION.md), submit a CSV, a public repository URL
and a short technical document by **17 November 2026**, using the form supplied
by email. No time of day is specified. A Google account and registration are
required; see [RULES.md](RULES.md) for team rules. No presentation or deployment
is required for this phase.

`data/`, `artifacts/`, `.cache/` and `.venv/` are excluded from Git. Match data and
reconstructable intermediate records must not be published or sent to hosted
LLMs. Submit the CSV only through the designated submission channel. The local
implementation publishes or sends nothing.

## Official challenge starter

The original instructions below describe the supplied tools.

# More than a Hack 2027 — Phase 1: Ball Possession

**Organised by [Metrica Sports](https://metrica-sports.com), [FC Barcelona](https://www.fcbarcelona.com)
and the [Mobile World Capital Barcelona](https://www.mwcbarcelona.com)**, at the Talent Arena in Barcelona.

A football data challenge on real professional match data, open to analysts, data scientists and
engineers. You are given one complete LaLiga match — two camera feeds, automatic tracking of every
player and the ball, automatic tactical tagging, and the camera-to-pitch geometry — and one
question to answer with it.

---

## The task

**For every frame of the match, say who has the ball.**

```
frame,label
11185,A        away in possession, ball alive
11186,A
...
18240,D        dead ball
...
18684,H        home in possession, ball alive
```

That is the whole task. One CSV, one row per video frame, 199,319 rows. It is scored automatically
against a private frame-by-frame ground truth, so there is a number at the end and no argument
about it — no jury, and no presentation to argue from.

The tracking is automatic and uncorrected: it has gaps and identity
switches, and the ball itself is invisible on 41% of the frames you are scored on.

---

## The challenge in one screen

| |                                                                                                                           |
|---|---------------------------------------------------------------------------------------------------------------------------|
| **Input** | one full match: tactical video, broadcast video, automatic tracking **with the ball** for both, Smart Tagging, homography |
| **Output** | `frame,label` CSV, 199,319 rows, labels `H` / `A` / `D` / `X`                                                             |
| **Ranked on** | `spellF1` — how many of the match's real possessions you recovered as whole possessions                                   |
| **Also reported** | `macroF1` — the per-frame view, and the first tie-break                                                                   |
| **Baseline to beat** | `spellF1` **0.2433**, in this repository                                                                                  |
| **Ground truth you get** | the first **5 minutes** of play — to check your pipeline, not to train on                                              |
| **Ground truth you are scored on** | the full match, **private**                                                                                               |

It is deliberately **not** a tracking challenge. The tracking, tagging and homography are given to
you so that nobody has to rebuild detection, re-identification and camera calibration before
getting to the actual question. See [CHALLENGE.md](CHALLENGE.md).

---

## Try it right now

The scorer, a 5-minute ground truth and four working baselines are in this repository. No
enrolment, no download, no data:

```bash
pip install -r requirements.txt
python evaluation/run_evaluation.py
```

Scored on the public 5 minutes, not on the full match -- which is why the order below is not the
order of the floor quoted above:

```
scoreboard  gt/BAR-ATH_possession_gt_public.csv  199319 frames  7500 graded  21 gt segments
ground truth has 17 possessions of mean 14.2 s  <- compare the poss and dur columns

rank  submission                macroF1  spellF1   poss     dur  
----  ------------------------  -------  -------  -----  ------  
   1  baseline_smooth125         0.5812   0.4242     16    18.5  
   2  baseline_smooth50          0.6207   0.3256     26    10.8  
   3  baseline_smooth25          0.6200   0.2868     36     7.6  
   4  baseline_raw               0.5972   0.0378    131     2.0  
```

Those four are **the same algorithm four times** — nearest tracked player to the ball, then a
majority vote over a window. Only the window changes. `macroF1` barely moves; `spellF1` moves by a
factor of eleven. That disagreement is the whole point of the metric, and
[EVALUATION.md](EVALUATION.md) explains it.

Once you have the data, regenerate them yourself and go further:

```bash
python starter/baseline.py                              # rebuild the four baselines
python evaluation/validate_submission.py <your>.csv     # before every submission
```

**Validate every file before you submit it.** `validate_submission.py` runs exactly the checks the
official scorer runs, and a file that fails them is dropped from the board with a one-line reason.
It takes two seconds and it is the single most common way a good entry scores nothing.

---

## Getting the data

The match assets are released to enrolled participants who have accepted the terms of use.

1. Enrol: **[More than A Hack 2027](https://landing.mobileworldcapital.com/more-than-a-hack-2027)**
2. Read and sign the terms and conditions in the registration form
3. You receive a download link.
4. Unpack it into `./data`, or point `MTAH_DATA_DIR` at wherever you put it.

Full description of every asset and how the frames line up: **[DATA.md](DATA.md)**.

---

## Read these, in this order

| | |
|---|---|
| **[CHALLENGE.md](CHALLENGE.md)** | the problem, the four labels, what counts as possession |
| **[DATA.md](DATA.md)** | every provided asset, its format, and the frame alignment |
| **[EVALUATION.md](EVALUATION.md)** | how you are scored, and why — read before optimising anything |
| **[SUBMISSION.md](SUBMISSION.md)** | the file contract, the three deliverables, how to submit |
| **[RULES.md](RULES.md)** | what you may and may not do |
| **[FAQ.md](FAQ.md)** | |
| **[OVERVIEW.md](OVERVIEW.md)** | the competition as a whole — context, why the 2027 format changed, the calendar |


---

## What is in this repository

```
games.py                    per-match constants, asset paths and the frame alignment
evaluation/
  possession_metrics.py     the metric library — every number on the board is defined here
  run_evaluation.py         the scorer. This exact file produces the official result
  validate_submission.py    check a file before you send it
  gt/                       the public 5-minute ground truth
  results/                  the four baselines; put your submissions here
starter/
  baseline.py               the reference baseline, numpy only
  load_atd.py               read the tracking and line it up with video frames
  load_smart_tagging.py     read the tactical tagging as video-frame intervals
  load_homography.py        read the homography; project pitch <-> pixels
  utils.py                  pitch plotting for the notebooks
  01_smart_tagging.ipynb    guided tour of the tactical tagging
  02_tracking_atd.ipynb     guided tour of the tracking data, with plots
  03_first_submission.ipynb data in, scored submission out — start here
data/                       unpack the downloaded match assets here. Empty in a fresh clone
```

The scorer you run locally is the scorer that decides the competition. If it scores on your
machine, it scores the same way on ours.

## Workspace layout for comparing our solution

The provided implementation and the reference outputs stay in their original locations so that
the supplied commands continue to work:

| Area | Contents |
|---|---|
| `starter/` | Provided loader code, notebooks, and the reference baseline |
| `evaluation/` | Official scorer, public ground truth, committed baseline submissions, and reference reports |
| `data/` | Downloaded match data used by both implementations |
| `solution/code/` | Our implementation |
| `solution/results/` | Our submission CSV files |
| `solution/reports/` | Reports generated from our submissions |

Use the solution-specific commands in [solution/README.md](solution/README.md) to score our work
without overwriting the reference scoreboard. This arrangement deliberately adds a comparison
area rather than moving `starter/` or `evaluation/`, because the supplied scripts use those paths.

---

## Requirements

**Python 3.8 or newer.** The evaluation needs `numpy` and `pandas` and nothing else; it is tested
under 3.8 and 3.11 and produces identical output under both. `starter/baseline.py` needs nothing
more either — you can reach a scored submission without installing anything else.

The notebooks additionally want `kloppy` and `matplotlib` (`requirements-starter.txt`), and
**Python 3.9+ is recommended for those**: on Windows, installing Jupyter under 3.8 hits a
`pywinpty` build failure that has nothing to do with this repository.

---

## Timeline

| |                |
|---|----------------|
| registration opens | **October 1, 2026** |
| data released | **October 20, 2026** |
| submission deadline | **November 17, 2026** |
| results announced | **Before November 26, 2026**      |

---

## Questions

Open an issue for anything about the data, the metrics or the documentation — corrections are
genuinely useful, and a fix that helps you helps everyone equally. For enrolment and the terms of
use: **hackathon.talentarena@mobileworldcapital.com**.

## Licence and citation

The code and documentation in this repository are released under the MIT Licence — see
[LICENSE](LICENSE). **The match data is not covered by it**: it is separately licensed for this
competition only, under the terms you accept at enrolment.

```bibtex
@misc{mtah2027,
  title  = {More than a Hack 2027: Phase 1 - Ball Possession},
  author = {Metrica Sports and FC Barcelona Innovation Hub},
  year   = {2027},
  url    = {https://github.com/metrica-sports/more-than-a-hack-2027}
}
```
