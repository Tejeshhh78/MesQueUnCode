# Progress and decisions

## Authoritative requirements

- Task: H/A/D/X for every tactical video frame, exactly 199,319 rows
  ([CHALLENGE.md](CHALLENGE.md), [SUBMISSION.md](SUBMISSION.md)).
- Ranking: spellF1; first tie-break macroF1; additional tie-breaks are defined in
  `evaluation/run_evaluation.py:165`. No subjective judging in Phase 1.
- Submission: team name, CSV, public repository and a short Markdown/PDF report;
  form supplied by email; deadline according to the repository: 17 November 2026.
  No time of day is specified.
- No manual match annotations, private or external human-produced match data,
  training or tuning on public GT (RULES.md).
- Raw data stays local; no raw records or frame-level extracts are sent to hosted
  LLMs (FAQ.md, OVERVIEW.md). No redistribution of assets.
- Reproduction must generate the same CSV. Data, local intermediate files and
  submission outputs stay out of Git.

## Initial findings

Unmodified starter with four baselines, the official scorer and public GT.
Tactical/broadcast tracking, homographies and tactical tagging are available
locally; videos are absent. No additional project agent instructions were found.
Python is not on PATH; the local Codex runtime contains Python 3.12.14,
NumPy 2.3.5 and pandas 3.0.1. No new services or paid APIs are needed.

## Strategy decision before inspecting GT results

| Approach | Benefit | Risk / decision |
|---|---|---|
| Nearest player + majority vote | Simple published reference | Confuses occlusions with stoppages; comparison only |
| Copy tagging directly | Useful hints when the ball is missing | Phases are not possession intervals; soft evidence only |
| Video model / broadcast fusion | Potential extra ball observations | Videos absent; broadcast unsynchronised; initially omitted |
| Geometry + tagging + temporal states | Uses complementary available signals | Selected; limitations without private GT explicitly unknown |

Internal priorities (not additional official scoring): valid submission and rule
compliance, reproducible functionality, stable possession intervals and stoppage
detection. No target possession count or possession share is derived from GT.

## Implementation and acceptance criteria

1. Strict streaming loader: validate complete frames, metadata and missing values;
   extract local geometry and movement signals.
2. Soft team/game-state evidence from tracking and automatic tagging; decode
   each half separately without carrying state across halftime.
3. Synthetic tests for ball gaps, deflections, controlled changes, stoppages,
   restarts, team assignments, invalid inputs and decoder optimality.
4. Complete local run; official validator; independent reproduction with SHA-256
   comparison. One public pipeline check only after fixing the method; no
   parameter selection from its results.
5. Short technical report, reproducible commands and final audit.

## Outstanding external items

Provisional team name from the repository: MesQueUnCode. Registration, 3–5 team
members, public repository URL, participation terms and form access are not
verified. No publication or submission without user authorisation.
Private full-match quality cannot be measured without private GT.

## Verification before the public pipeline check

- 13 synthetic tests passed, including exhaustive comparison of the decoder with
  all paths of a small problem (minimum durations 1/2/3).
- Complete local run: 199,319 frames strictly read and output; official validator
  passed. 210 predicted possession spells (not a statement about their correctness).
  First run: 29.38 s on the available runtime.
- Until this point, the solution had neither read public GT nor used it for
  parameter decisions. The method and Config in `possession/model.py` were fixed
  for the one-time pipeline check; its results were not used to compare variants.
- Pinned packages were already installed locally; `pip install --no-index`
  confirmed the available versions. A download into a completely empty
  environment had not been checked.

## One-time public check

After fixing the method: official scorer, 7,500 public frames and 1,000 bootstrap
draws. spellF1 0.444444 (5th–95th percentile 0.237037–0.651852), macroF1 0.736204
(0.570675–0.813174). No parameter changes or variant selection based on these
results. No claims about the unknown private score.

## Final requirements audit

| Requirement | Status / evidence |
|---|---|
| Algorithmic full-match H/A/D/X output | Verified; 199,319 frames; official validator |
| Frame alignment and valid labels | Verified; strict loader, tests, official validator |
| Implement possession/dead-ball rules | Implemented and tested synthetically; semantic full-match correctness unknown without private GT |
| Exact regeneration | Verified; independent complete runs, identical CSV bytes and SHA-256 |
| Executable, inspectable source code | Verified locally; `python -m possession` |
| Pinned dependencies and execution instructions | Available; packages confirmed locally; completely fresh installation not verified |
| No manual labels or corrections | Satisfied by the developed pipeline; no rules tied to individual frames |
| No GT training or tuning | Satisfied; one public check after fixing the method |
| No private or external human-produced match data | Satisfied by the developed pipeline; only permitted automatic assets |
| No data publication | Checked locally; data, outputs and environment ignored; nothing published externally |
| Short technical report | `TECHNICAL_REPORT.md`; copied to the local output as team_name.md |
| Public repository URL | Outstanding; no push or visibility change without authorisation |
| Registered team name / 3–5 members | Outstanding; MesQueUnCode is a filename assumption |
| Form / Google account / actual submission | External organisational steps outstanding |
| Deadline and format | According to SUBMISSION.md: 17 November 2026, CSV + public URL + Markdown/PDF; no time of day specified |
| Private spellF1 > 0.2433 / winning | Not demonstrated; cannot be checked without private GT |

## Critical competition assessment

Strengths: a missing ball is not automatically D; multiple automatic signals;
explicit temporal structure; exact decoder checks and demonstrated reproduction.
Main risks: short genuine possessions can be merged; fouls with players still
moving are hard to detect without video; tagging errors have more influence
during long ball gaps. General superiority cannot be inferred from 17 public
possession spells. Useful continuation: obtain permitted labels from other
matches for independent validation, improve stoppage and short-possession
detection there, then consider local video evidence.

The local project is prepared for technical handover. Formal submission still
depends on the external information above and publication of the source code.
CSV files and match data are not pushed publicly.

## Runtime comparison on 26 October 2026

At the user's request, `benchmark_runtime.py --runs 3` performed one warm-up per
method followed by three sequential measured runs with alternating order.
The complete CLI call is measured, including process startup, reading and writing.
Our pipeline also includes its validation and manifest generation. No GT is read;
all outputs were valid and byte-identical across repeated runs of each method.

| Method | Median | Minimum–maximum |
|---|---:|---:|
| Original baseline_smooth50 | 4.88 s | 3.91–4.97 s |
| Our solution | 26.74 s | 21.88–36.43 s |

Our complete pipeline took 5.49 times as long in this local comparison. It was
therefore not faster than the baseline. Runtime varies with machine load; this
is not an isolated benchmark of the algorithm core. Individual measurements
were saved in `artifacts/runtime-benchmark/timings.json`; that original series
was subsequently archived as `timings-before-optimization.json`. See README.md
for repetition instructions.

## Implementation optimisation on 26 October 2026

Following the runtime concern, the loader and decoder were profiled. The first
profile attributed 39.04 s to the loader and 4.56 s to the decoder. Profiled
runtimes are not equivalent to unprofiled CLI measurements.

- Process blocks of at most 4,096 frames using a strict NumPy text parser and
  vectorised team/movement features. Preserve the original reduction order for
  mean speeds.
- Use small float lists and three bytes of predecessor information per frame
  in the sequential Viterbi recurrence instead of many temporary NumPy arrays.
  Minimum duration and tie-breaking rules remain unchanged.
- Compare the decoder with the old implementation on 42 synthetic cases and
  80,000 synthetic frames: exactly identical paths. The separate 80,000-frame
  run took 2.02 s before and 0.47 s after optimisation.
- 16 tests passed, including new checks for block-size invariance, incomplete
  numeric fields and decoder ties. No new GT evaluation or inference-parameter
  changes.
- Full-match output remained byte-identical across all repeated optimised runs;
  SHA-256 remains `3ce21a4bf873fcfe7a04013408537d230e8a32d361ea18ec430d6e12c77bd43e`.

Updated measurement series using the same benchmark protocol:

| Method | Median | Minimum–maximum |
|---|---:|---:|
| Original baseline_smooth50 | 5.62 s | 4.09–8.13 s |
| Optimised solution | 13.45 s | 12.45–23.07 s |

The complete optimised pipeline now takes 2.39 times as long as the baseline.
Its observed median is roughly half the previous series, but machine load varies,
so this is not an isolated hardware benchmark. The baseline remains faster.
The old series is in `artifacts/runtime-benchmark/timings-before-optimization.json`;
`timings.json` contains the new one. Public scores remain unchanged because the
CSV is identical; private full-match superiority is still not demonstrated.

## English consistency update

Translated all added documentation, comments, docstrings, CLI messages and local
helper scripts into English. Renamed the report to `TECHNICAL_REPORT.md` and
updated references and generated report copies. Inference parameters and
possession-label semantics are unchanged.
