# Progress and decisions

Paths and commands recorded before 27 October 2026 refer to the original layout.
See [README.md](README.md) for current commands under `solution/`.

## Authoritative requirements

- Task: H/A/D/X for every tactical video frame, exactly 199,319 rows
  ([CHALLENGE.md](../docs/CHALLENGE.md), [SUBMISSION.md](../docs/SUBMISSION.md)).
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

## Fixed head-to-head diagnostic on 26 October 2026

At the user's request, `compare_algorithms.py --runs 3 --bootstrap 1000`
regenerated both the preselected original `baseline_smooth50` and the unchanged
current solution. It reused the runtime benchmark (one warm-up and three measured
runs each, sequential execution, alternating order), then evaluated exactly those
two CSVs with the unchanged official public scorer. Input checksums were identical
before and after the test; every output passed official validation and was stable
across repeated runs. Both methods use the same match and tactical tracking; our
method also uses the supplied automatic Smart Tagging.

| Metric | Original baseline | Current solution |
|---|---:|---:|
| Complete-match median CLI runtime | 6.51 s | 12.76 s |
| Public spellF1 | 0.3256 | 0.4444 |
| Public macroF1 | 0.6207 | 0.7362 |
| Public frame accuracy | 0.7132 | 0.7557 |
| Public dead-ball F1 | 0.3225 | 0.6652 |
| Public dead-ball recall | 0.2138 | 0.5221 |
| Public predicted possession count; reference 17 | 26 | 10 |

Runtime covers all 199,319 frames; quality covers only the 7,500 public frames.
The runtime ratio in this series is 1.96. The official spellF1 bootstrap intervals
(5th–95th percentiles, 1,000 draws) are 0.2140–0.4465 for the baseline and
0.2370–0.6519 for our method. They overlap substantially and are not a paired
significance test. The higher public point estimate does not establish private
full-match superiority. The solution still predicts too few possession intervals
in this window, while the baseline predicts too many. No model parameters or
method choices were changed using this diagnostic.

Aggregate comparison reports are local at `artifacts/head-to-head/comparison.md`
and `comparison.json`; the official scoreboard and reports are in the adjacent
`evaluation/` directory. The original outputs remain out of Git. All new code and
documentation are in English.

## Possession quality comparison on 26 October 2026

Extended `compare_algorithms.py` with a dedicated possession-quality table and
local `possession_quality.md` / `possession_quality.json` reports. They separate
the official mean spellF1 from F1 at a single IoU threshold and expose matched,
missed and unmatched phases, precision, recall and all official IoU thresholds.
`--quality-only` verifies saved CSV checksums against the recorded runtime run,
then invokes the unchanged official public scorer without inference or timing.
Reports explicitly identify reused versus freshly generated predictions.

At IoU >= 0.5, each method matches 7 of 17 reference phases and misses 10.
The new method has 3 unmatched predictions versus the baseline's 19: precision
is 70.00% versus 26.92%, with recall unchanged at 41.18%. F1 at this threshold
is 0.5185 versus 0.3256; the official mean across thresholds is 0.4444 versus
0.3256. Equal match counts do not imply that the same reference phases matched.
The gain at this threshold is precision, not additional recovered phases.
At the looser IoU thresholds 0.3 and 0.4, the baseline recalls more phases;
at the stricter thresholds 0.6 and 0.7, the new method matches one more phase.
Only the public five-minute window is scored; private performance remains unknown.

Verification: the quality-only command completed with 1,000 bootstrap draws.
The generated JSON exactly preserves the official results and prediction hashes.
CLI checks confirmed rejection of missing run provenance, missing predictions,
altered predictions and additional CSV submissions before scoring. A complete
integration run in `artifacts/head-to-head-smoke` regenerated both predictions
with one measured run, no warm-up and no bootstrap; scoring and both report types
completed successfully. This was a functional check, not a new runtime benchmark
series. Existing head-to-head runtime records were preserved. No inference code,
model parameters, match labels or official scoring code were changed.

## Team algorithm guide on 26 October 2026

Added `ALGORITHM_GUIDE.md` and linked it from the README. The English guide explains
the baseline, input features, evidence equations, movement fitting, temporal
decoding, official metrics, measured limitations and a rule-compliant development
plan. It distinguishes implementation facts, measurements, hypotheses and proposals.
Its two executable teaching examples use entirely synthetic inputs; their output
was checked against the documented values. All 49 internal links and the README
link were checked. Inference and evaluator code remain unchanged.

## Integrate the shared layout on 27 October 2026

Fetched `origin` and merged `origin/testing` at `3fefcc1` into the local `dev_VG`
branch. No remote branch was pushed. The shared layout places official challenge
documents in `docs/` and our implementation in lowercase `solution/code/`.

Moved the possession package, comparison tools, synthetic tests, pinned solution
dependencies and team documentation under `solution/`. Updated imports, executable
entry points, source checksums, report copying and documented commands. Default
submissions now go to `solution/results/`; comparisons and timing reports go to
`solution/reports/`. Both generated directories are excluded from Git. Existing
local `artifacts/` outputs remain available. The team branch removes committed
baseline CSVs; the comparison still regenerates its baseline from the shared data.

Verification completed:

- All 16 synthetic tests passed from `solution/tests/`.
- Default inference wrote and validated 199,319 rows under `solution/results/`.
- The CSV SHA-256 remained
  `3ce21a4bf873fcfe7a04013408537d230e8a32d361ea18ec430d6e12c77bd43e`.
- Manifest source paths and the copied technical report match their new locations.
- A full comparison with one run, no warm-up and no bootstrap completed from the
  new module entry point. This was an integration check, not a benchmark series.
- The direct-script `--quality-only --bootstrap 1000` path also completed;
  spellF1 remains 0.3256 for the baseline and 0.4444 for the unchanged solution.
- Direct comparison/timing entry points import successfully outside the repository
  root, and the guide's relocated synthetic examples match their expected output.
- All 114 checked documentation link targets exist, and generated output files
  remain ignored by Git.

The model, inference parameters and official scoring code were not changed.

## Quality improvement masterprompt on 27 October 2026

Reviewed `dev_VG` at `35c4dc6`, the current inference and comparison code, synthetic
tests, official challenge/rule/evaluation/submission documents, asset availability
and existing aggregate results. Added `QUALITY_IMPROVEMENT_PROMPT.md`, linked from
both READMEs, as an executable engineering brief for a future development run.

The prompt prioritises independent other-match validation, local evidence
diagnostics and controlled experiments on observation reliability, ball control,
stoppages, tags and temporal structure. It explicitly prevents public-GT-driven
training, selection and rollback, and distinguishes competition restrictions from
conservative project acceptance policies. It preserves an incumbent when real-match
improvement cannot be established and requires honest reporting of that limitation.

Checked all 24 source links, both README references, command entry-point paths and
Markdown fences. This task changed documentation only; no inference, new model
selection, public rescoring, commit or push was performed.

## Execute independent quality development on 27–28 October 2026

Executed the masterprompt on `dev_VG` from `35c4dc6`, preserving all existing
uncommitted documentation. No branch change, commit, push, public competition
rescoring or submission was performed. Official challenge/evaluation source and
references were unchanged. Competition public labels were not used for
candidate selection, rollback or parameter fitting.

Frozen incumbent source and output are under
`solution/reports/quality-development/incumbent-35c4dc6/`. The CSV was reproduced
with diagnostics and again after all changes:
SHA-256 `3ce21a4bf873fcfe7a04013408537d230e8a32d361ea18ec430d6e12c77bd43e`.

Implemented optional aggregate diagnostics and an explicit, Git-ignore-checked
local NPZ trace. Added independent-match evaluation using the unchanged official
scorer, explicit provenance/units/25-Hz contracts, complete-match splits and
separate input/reference channels. Review caught a period concatenation bug
before real candidate evaluation: the scoring grid now inserts 51 ignored
X frames only between metadata periods, preserving all original frames and
within-period timing. This prevents halftime possession merging and cross-period
transition matches. Inference remains on the source timeline.

Researched lawful other-match data and selected the public DFL/IDSSE release
by Bassek et al. (2025), CC BY 4.0. Imported four games at native 25 Hz with
streaming XML processing, physical unit conversion and separate provider-reference
CSVs. Downloads were verified against published MD5s; manifests record SHA-256s
and provenance. Target-field mutation tests verify that reference changes cannot
affect features or period construction. Attribution, source URLs and reproduction
are in [INDEPENDENT_VALIDATION.md](INDEPENDENT_VALIDATION.md).

Two protocols were written before their candidate evaluations and subsequently
copied byte-identically into tracked `solution/experiments/` for reproduction:

1. **motion-reliability-001:** isolated ball jumps rejected using the incumbent's
   45 m/s plausibility bound. A synthetic failure improves, but J03WMX pilot
   predictions are identical to incumbent: spellF1 0.374570, macroF1 0.856095.
   No measured natural-match benefit; profile remains experimental.
2. **control-switch-cost-002:** evidence, minimum duration and stoppage costs
   retained; only H/A switch costs 4/8/12 compared with 20. Development matches
   J03WMX/J03WOH; final J03WN1/J03WOY reserved before candidate scores. Cost 4
   failed primary/secondary gates. Cost 8 had the best primary development mean
   but failed the per-match macroF1 limit. Cost **12** alone passed all gates:
   development mean 0.400188 -> 0.417815. Frozen cost 12 then passed all gates
   on the once-consumed final set: **0.354710 -> 0.389586**, both games improving.

The final improvement is +0.034876 absolute, approximately +9.83% relative.
At IoU 0.5, correct matches increase from 252 to 305, while unmatched predictions
rise from 221 to 285: a precision/recall trade-off. Final macroF1 changes
(+0.000758/-0.002357) remain within the declared limit. Single-pass runtime
ratios are noisy (1.467/0.936); no speedup claim is made. Full metrics and hashes
remain in ignored `control-switch-cost-002/` and `control-switch-cost-002-final/`.

The driver records source/manifest/protocol/configuration hashes and checks
regression limits per match. Shared dataset markers prevent an accidental
second grid or final run through another output directory. These safeguards
do not prove independence or protect against tampered provenance.

**Decision:** expose responsive cost 12 as the independently supported candidate,
while retaining incumbent as CLI default and exact reproduction reference.
After final testing, changed only the responsive convenience default from 8 to
the already selected 12. Evaluated sources are archived in `evaluated-source/`
under the final report. Both profiles' final-game prediction hashes were then
reproduced exactly without opening reference labels or recomputing scores.
No final-result-driven retuning occurred.

Verification completed:

- All **90 tests passed**, including original decoder/data checks, tracing,
  both experiments, independent adapters and selection/final safeguards.
  No tests were skipped.
- `pip check`: no broken requirements; no new runtime dependencies.
- Two full competition-data candidate runs (explicit cost 12 and its matching
  profile default) produced identical 199,319-row CSVs, SHA-256
  `368f50cd5d6d826b97c80e1d492dd474392429ed1430ce60ead243c62d1ebea2`.
- The candidate passes the unchanged official format validator. Full CLI times
  were 17.42 s and 13.11 s; incumbent recheck 10.25 s. These are reproduction
  observations, not an alternating controlled benchmark or speed guarantee.
- Optional traces and downloaded/generated artifacts remain ignored by Git.
- Final local audit confirms half-boundary/X coverage, matching manifest source
  hashes and 106 valid local documentation links; `git diff --check` is clean.
- Updated README, technical report, guide scope and independent-validation
  document. Preserved the masterprompt as the historical instruction.

**Limits and next step:** two final games cannot establish statistical
significance or the private competition score. External tracking is cleaner
than competition ATD, and development/final games share Fortuna Duesseldorf:
match-disjoint, not team-disjoint evidence. Do not retune on the consumed final
matches. Further selection requires a predeclared hypothesis and new held-out
games with inspectable control/stoppage labels, preferably from an observation
domain closer to the supplied tracking. Do not use the public competition window
as a substitute. The separate candidate is ready for review; publishing and
submission remain unperformed.

## Fixed candidate versus original baseline on 29 October 2026

At the user's request, compared the independently selected responsive cost 12
against the original `baseline_smooth50` on identical supplied tracking. Added
explicit `--profile` forwarding to comparison/runtime tools and profile/config
metadata in reports. A regression test rejects saved incumbent predictions
being re-scored under the responsive name, before any scoring occurs.

Executed one warm-up and three alternating measured full CLI runs per method,
then one unchanged official public evaluation with 1,000 bootstrap draws.
The candidate CSV hash remains the frozen `368f50cd...1ebea2`; no inference
weights, variant choice or default changed in response to this check.

- Mean public spellF1: baseline 0.3256, responsive 0.4138 (+27.1% relative).
- MacroF1: 0.6207 -> 0.7278. At IoU 0.5 both recover 7/17 possessions; unmatched
  predictions decrease from 19 to 5. Precision improves; recall is unchanged.
- Median full CLI time: baseline 4.5314 s, responsive 10.6895 s (2.36 times as
  long). Individual measured candidate times: 11.14, 10.58, 10.69 s.
- SpellF1 5th–95th bootstrap ranges: baseline 0.2140–0.4465, candidate
  0.2207–0.6076. These separate, overlapping intervals are not a paired test.
- All 91 tests pass. Exact profile/configuration and frozen prediction hash
  were verified in generated reports. Generated data and reports remain ignored.

Only 7,500 public frames/17 phases are graded; full-match private quality is
unknown. The incumbent's earlier public score of 0.4444 does not change the
independent development/final-test decision. The public result is neither a new
selection objective nor a rollback criterion.

Reports: `solution/reports/head-to-head-responsive-2026-10-29/`, including
`comparison.md`, `comparison.json`, `possession_quality.md` and runtime records.
README and technical report distinguish original baseline, incumbent and
selected responsive profile. Source/docs/tests are prepared for the separately
requested push to `origin/dev_VG`; raw data and generated artifacts are excluded.
