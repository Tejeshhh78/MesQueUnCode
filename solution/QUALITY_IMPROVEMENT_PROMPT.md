# Masterprompt: improve possession detection with verifiable evidence

Prepared on 27 October 2026 after reviewing local `dev_VG` at `35c4dc6`.
This is an instruction for a future development run, not a claim that the proposed
improvements have been implemented. Read the current repository again when using
it: the snapshot below may become outdated.

To use it, ask your engineering agent to read this entire file and execute the
masterprompt from the repository root. The scope includes implementation and
validation; it excludes publishing, pushing or submitting the project.

---

## Mission

You are responsible for improving the quality of this repository's football
possession inference system. Work as an experienced temporal-modelling engineer,
data scientist and critical reviewer. Build and test meaningful improvements;
do not stop after writing an analysis or proposing a plan.

The objective is the strongest **rule-compliant, reproducible possession detector
supported by legitimate evidence**. Prioritise complete possession phases, correct
control changes, stoppages and restarts. Complexity, code volume and attractive
public scores are not objectives. Do not promise a particular improvement before
measuring it on permissible independent data.

Explain findings and progress to the user in German. Keep source code, comments,
CLI messages, repository documentation and experiment records in English.

## 1. Establish the current facts before changing anything

Start with `git status`, the active branch, recent history and any applicable
`AGENTS.md`. Preserve user changes and the current working implementation.
Do not reset, overwrite or silently discard existing work.

Read the following sources, resolving disagreements against the actual official
scoring code for metric definitions and against the official rules for eligibility:

- [Challenge](../docs/CHALLENGE.md), [rules](../docs/RULES.md),
  [FAQ](../docs/FAQ.md), [evaluation](../docs/EVALUATION.md),
  [data](../docs/DATA.md) and [submission](../docs/SUBMISSION.md).
- [Solution instructions](README.md), [algorithm guide](ALGORITHM_GUIDE.md),
  [technical report](TECHNICAL_REPORT.md) and [worklog](WORKLOG.md).
- [Data extraction](code/possession/data.py), [model](code/possession/model.py),
  [CLI and manifest](code/possession/__main__.py), [tests](tests/test_possession.py),
  [comparison](code/compare_algorithms.py) and [runtime benchmark](code/benchmark_runtime.py).
- [Original baseline](../starter/baseline.py), [shared metadata](../games.py),
  [official metrics](../evaluation/possession_metrics.py),
  [scorer](../evaluation/run_evaluation.py) and [validator](../evaluation/validate_submission.py).

Inspect relevant dependencies, configuration and existing experiment tooling.
Inventory asset availability without printing restricted file contents. Verify
which independent datasets are actually available; do not invent access to them.
Existing aggregate reports can establish what was previously checked, but cannot
become a model-selection objective when derived from the competition reference.

The reviewed starting point is:

| Area | Observed state; verify before relying on it |
|---|---|
| Branch/layout | `dev_VG`; implementation under `solution/code/`, tests under `solution/tests/` |
| Method | Geometry, automatic tagging, unsupervised movement mixture, three-state Viterbi with minimum dwell time |
| Temporal assumptions | Team-switch cost 20, stoppage-switch cost 24, minimum duration 0.32 s; defined in `Config` |
| Inputs used | Tactical tracking, its metadata and automatic Smart Tagging |
| Available locally at review | Tactical/broadcast tracking and homographies; tactical tags; neither video file present |
| Features computed but unused | Closest-player speed per team, fast-player fraction and visible-player count |
| Tests | 16 synthetic tests; no independent labelled-match evaluation harness found on this branch |
| Existing comparison | Fixed `baseline_smooth50` versus current solution; public check only, not evidence of private full-match superiority |
| Submission | 199,319 rows; current frozen CSV SHA-256 below |

```text
3ce21a4bf873fcfe7a04013408537d230e8a32d361ea18ec430d6e12c77bd43e
```

This checksum identifies the reviewed incumbent's output. It is not the required
output of a legitimately improved model. First verify the incumbent can reproduce
it; record a separate checksum for every later candidate.

Keep two reference methods conceptually separate: the published nearest-player
baseline and our current incumbent. Improving on the former alone does not prove
an improvement over the latter.

## 2. Non-negotiable competition constraints

Apply these throughout research, implementation, testing and reporting:

1. **No training, tuning or variant selection on the public five minutes.** This
   includes thresholds, weights, smoothing, architecture, feature selection,
   early stopping, ensembles and promotion or rollback decisions. Previously
   known scores, error lists, phase counts and durations are not alternative
   tuning signals. Splitting those minutes or bootstrapping them does not change
   the restriction. Do not optimise toward a known possession count or public score.
2. **No human labels or corrections of this competition match.** Do not annotate
   transitions, repair tracking by hand, insert timestamp-specific rules or patch
   predicted labels. Inspection for understanding general phenomena is allowed;
   turning an inspected scene into a special-case answer is not.
3. **No private ground truth or external human-produced information about this
   same game.** This includes corrected tracking, event feeds and possession labels,
   even when a source is publicly accessible. Do not search for a leaked answer key.
4. **Respect the allowed alternatives.** Labels from other games may be used when
   accessible to the organisers and documented. Unsupervised/self-supervised
   learning and wholly algorithmic pseudo-labels on the supplied automatic data
   are permitted. Public reference labels cannot be relabelled as pseudo-labels;
   agreement with our own predictions or Smart Tagging is not independent accuracy.
5. **Protect restricted assets.** Do not transmit competition videos, datasets,
   confidential material or extracts to hosted/cloud LLMs. Do not expose raw
   tracking, frame-level labels, event extracts or match screenshots in tool output
   to a hosted assistant. Local code may process them; return only safe aggregate
   checks. Do not publish assets or reconstructable derivatives through Git,
   notebooks, diagnostics or reports. Follow the actual data terms as well as the
   repository's MIT licence; they are different permissions.
   Default to local processing as a conservative project policy. The cited rules
   specifically prohibit hosted/cloud-LLM disclosure and restricted redistribution;
   do not misstate them as a blanket ban on every kind of external non-LLM
   computation. Any other processing still needs to comply with the actual data
   terms and the user's access/cost authorisations.
6. **Use inspectable resources.** For external datasets/models, verify the source,
   licence, access conditions, annotation provenance and compatibility before use.
   Research general methods with primary sources; never include restricted match
   content in web queries. Cite the resources actually used. Do not make private
   internal datasets or unavailable models a required dependency.
7. **Preserve the official evaluator and reference.** Do not change them to improve
   results. If a genuine evaluator defect is suspected, isolate the observation,
   document it and draft a clarification. Do not silently repair the answer key or
   treat an easier custom metric as the official score.
8. **Regenerate the exact submitted CSV.** Preserve frame alignment, accepted labels,
   deterministic execution and accessible dependencies. A postprocessed file that
   cannot be recreated by the documented pipeline is not an acceptable submission.

The official restrictions above come from the challenge documents, especially
[RULES.md](../docs/RULES.md), [FAQ.md](../docs/FAQ.md) and
[EVALUATION.md](../docs/EVALUATION.md). Distinguish them from the conservative
project policies explicitly marked here and below. If a proposed dependency or experiment genuinely requires
a rule interpretation, pause that part and prepare the precise question for the
user/organisers. Continue unrelated permitted work. Do not contact anyone yourself.

## 3. Optimise the right behaviour

The official primary score is `spell_f1_mean`: the mean possession F1 over temporal
IoU thresholds 0.3, 0.4, 0.5, 0.6 and 0.7. Matches require the same team and a
one-to-one assignment. A possession is a maximal `H` or `A` interval. `D` is a
stoppage and separates two possessions even when the same team owns both.

Controlled possession matters. A nearby player, a deflection or an aerial touch
does not by itself establish a new owner. During contested/loose play, the previous
team retains possession until the other team gains control. Missing ball tracking
is not equivalent to dead ball.

Use these diagnostics together on permissible validation data:

- Mean spellF1, plus TP/FP/FN, precision, recall and F1 at every official IoU.
- Frame macroF1 and per-class H/A/D precision and recall.
- Transition detection, matched-transition counts and absolute boundary error.
- False splits, merged phases and results across short/long phases and missingness.
- Whole-pipeline runtime and memory under documented conditions.

Phase count, mean duration, frame accuracy, confidence or smoothness alone cannot
establish success. `X` does not hide graded frames: macroF1 charges it, while
temporal scoring fills it inside reference-valid blocks. Do not use abstention as
a way to game evaluation.

Runtime is secondary to detection quality here; it is not an official ranking
component. Use practical compute limits and avoid unnecessary cost, but do not
force online/causal inference when this offline challenge allows future evidence.

## 4. Freeze the incumbent and establish the evidence protocol

Before any behaviour-changing edit:

1. Record source revision, working-tree state, configuration, environment, data
   checksums and incumbent output hash. Preserve the incumbent's files and commands.
   Place generated snapshots and restricted diagnostics under ignored output paths.
2. Run the existing synthetic suite and validate a regenerated incumbent CSV when
   inputs and dependencies are available. Explain any discrepancy before proceeding.
   Inspect existing public aggregates instead of repeatedly re-scoring them.
3. Separate software correctness from model quality. Synthetic tests and unchanged
   output establish specific properties, not generalisation to real matches.
4. Search for an existing compatible independent-data harness before building one.
   Audit its labels, splits, units, frame rate and metrics; do not assume another
   implementation is trustworthy just because it already exists.
5. Identify lawful labelled **other matches** for development. Check that controlled
   possession and dead-ball labels mean the same thing. Sparse event annotations
   are not automatically frame-exact possession ground truth. Document any derived
   labels and uncertainty rather than presenting proxy labels as authoritative.
6. Split by complete matches, keeping repeated views/crops of a match in the same
   split. Fit supervised parameters on training matches, select on development
   matches and reserve a final test set. Do not leak validation/test labels through
   normalisation, pseudo-label generation, tracking alignment or feature creation.
   If using permitted label-free adaptation at inference, declare that protocol and
   apply it consistently to both development and final evaluation.
7. Before candidate results are examined, write the primary aggregation, secondary
   regression limits, practical improvement threshold and stopping rule. Explain
   them as internal engineering choices, not official hackathon requirements.
   Report per-match values so pooling cannot hide failures on individual games.

The current inference entry point is tied to `games.GAME`, even though the core
`predict` function accepts features, events, blocks, pitch and fps. If needed,
create a small independent-match adapter around that core. The official metrics
use frame-based tolerances and a 25 fps duration convention: use an explicit,
verified time-base adapter for other frame rates. Do not pad, trim or fabricate
labels to make arbitrary external data resemble the competition input.

Keep competition data outside supervised training, hyperparameter search,
independent validation and final test datasets. General automatic adaptation to
its unlabelled tracking remains permitted under the documented inference protocol.

## 5. First build observability and meaningful correctness checks

Add only diagnostics that help answer specific hypotheses. Useful local signals
include observation validity, geometric/team evidence, tag contribution, movement
evidence, live/dead score, emission margins, state duration and transition penalties.
These are scores, not calibrated probabilities unless calibration is actually shown.

Ensure diagnostics do not alter inference. Store restricted traces locally and
provide safe aggregate summaries to the assistant. Avoid exposing exact match
timelines, screenshots or frame-level records in conversation or tracked reports.

Extend the synthetic suite with general, independently invented situations:

- A brief deflection versus a short but genuinely controlled `H -> A -> H` phase.
- A missing-ball interval with constant possession versus one with a genuine
  change and supporting observable context on either side.
- A stationary live ball versus a stationary dead ball; moving players during a
  stoppage; a real restart with incomplete or inaccurate automatic tags.
- Contradictory, duplicate, missing or misidentified team tags.
- A missing team, duplicated tracklets, invalid speeds, implausible ball jumps and
  very sparse observations.
- Half boundaries, frame/time alignment, deterministic ties and numerical stability.
- Appropriate metamorphic checks, such as swapping team identities away from
  exact ties, preserving physical positions under valid unit conversions, and
  invariance to the input block size.

Vary independently specified durations, noise and missingness rather than matching
known competition scenes or counts. Add tests for competing failure modes, not only
for the behaviour a new feature is designed to favour. Do not demand magical exact
recovery of a completely unobservable transition; state what evidence or uncertainty
bound makes the expected result identifiable.

Keep the existing exhaustive small-problem decoder check: a correct optimiser
for a poor model and an incorrect optimiser for a good model are distinct problems.

## 6. Investigate high-value mechanisms, not an arbitrary architecture

Evaluate the following hypotheses in priority order, revising the order when
independent evidence justifies it. These are research candidates, not instructions
to implement all of them or claims of improvement.

| Mechanism | Limitation visible in the current code | Candidate direction | Important counter-risk |
|---|---|---|---|
| Observation reliability | Both finite team distances are required; several visibility features are unused | Confidence-aware geometry and robust collective movement using valid counts/distributions | Missingness and duplicate tracks can introduce team bias |
| Actual ball control | Nearest-team distance and scalar ball speed are weak control proxies | Sustained local control, relative motion, acceleration/direction changes, reliable short tracklets | Track slots are not player identities; a fast pass or aerial ball is not a turnover |
| Stoppages and restarts | Movement regimes are not semantic live/dead classes; restart fallback can be weak | Combine observation reliability, movement distributions, ball evidence and soft restart cues | Slow live play and running after a whistle must both remain possible |
| Tactical tags | Tags can be correlated, wrong or temporally imprecise | Bounded support, disagreement handling and reliability checks from independent data | Letting tags dictate labels can create long incorrect possessions |
| Temporal structure | Fixed switch costs and minimum dwell time cannot adapt to evidence quality | Evidence-dependent transitions or an explicit duration model | Lower penalties can restore flicker; higher penalties can erase true short phases |
| Learned evidence | Current heuristic scores are not calibrated | Simple interpretable classifiers or temporal models trained on compatible other matches | Label mismatch, leakage and domain shift can outweigh apparent gains |

Start with the simplest mechanism that addresses an independently reproduced
failure. Existing unused features are opportunities, not proof that using them
will help. Do not equate the faster Gaussian component with ground-truth live play.
Do not assume reducing switch costs is the answer merely because a public report
contains fewer predicted possessions.

Normalising the same emission potentials with a per-frame log-softmax only adds
a common frame-dependent constant to all candidate paths; it does not improve
the decoded sequence by itself. Treat calibrated uncertainty as useful only when
its calibration and downstream decision rule are independently evaluated.

More advanced sequence models, ensembles or local video evidence are optional
only when data, licensing, compute and independent validation justify them. At the
reviewed snapshot videos are absent, so do not base the critical path on video
processing. Broadcast fusion requires an algorithmic timeline mapping and handling
cuts/replays; tactical and broadcast frame indices are not interchangeable.

The challenge allows any inspectable method. Prefer a stronger simple model over
an unsupported complex one; remain willing to change architecture when evidence
supports the added cost.

## 7. Implement and compare controlled experiments

For each selected experiment:

1. Record the hypothesis, code evidence, exact change, expected benefit, expected
   regression and permissible data source before evaluating the candidate.
2. Implement the change in `solution/` with a reversible way to run the incumbent
   and candidate. Keep defaults and output directories explicit. Do not accidentally
   overwrite incumbent manifests, predictions or reports.
3. Run relevant synthetic/regression checks. Fix software failures before making
   quality claims. Do not change tests simply to endorse the new implementation.
4. Run incumbent and candidate on the same independent matches with identical
   time alignment, labels and evaluation settings. Use the original baseline as
   an additional reference when its input contract can be matched faithfully.
5. Report the official metrics per match and in aggregate, with paired differences.
   Use uncertainty appropriate to independent games and temporal correlation;
   do not treat frames as independent samples. Few games require cautious claims.
   Independent intervals around two methods are not a paired test of their gain.
6. Use targeted ablations on development data to isolate a component's contribution.
   Track failed and neutral experiments as well as successful ones. Do not turn
   the final held-out test into another iterative development set.
7. Measure performance after correctness and quality. Separate whole-CLI timing
   from profiling of parsing, evidence, decoding and output. Treat an intentionally
   changed model separately from implementation-only optimisations.

Maintain a compact experiment record with revision, config/seed, input and output
hashes, dataset/split provenance, commands, results and the accept/reject reason.
Use `solution/WORKLOG.md` for decisions and ignored `solution/reports/` for generated
reports. Do not proliferate documents or dependencies without a clear purpose.

**Tooling trap:** `solution.code.compare_algorithms` always invokes the competition's
public scorer. It is not an independent development harness. `--quality-only`
re-scores saved predictions; it does not generate predictions from changed code.
Neither mode may become an automated candidate-selection loop. Build or reuse a
separate independent-match harness for that purpose.

## 8. Decide what can become the default

The following are conservative **project acceptance policies**, not additional
competition rules:

- Keep the incumbent runnable until a candidate meets the predeclared independent
  validation criteria and passes all mandatory correctness/reproduction checks.
- A general software defect may be fixed based on independent tests and a clear
  contract violation. Document its effect; do not call that a measured football
  quality gain without supporting evaluation.
- For implementation-only changes, require byte-identical predictions on the
  same inputs/configuration. For intended model changes, record new output hashes;
  demanding the old hash would prevent any improvement.
- Do not promote or roll back a candidate because its public competition score
  increased or decreased. Public results cannot break an independent-data tie.
- If evidence is weak, mixed or underpowered, report that and keep the candidate
  experimental. A slightly higher point estimate does not automatically establish
  a meaningful improvement.

If suitable independent labels are unavailable, continue useful work: local
diagnostics, verified adapters, synthetic/metamorphic tests and permissible
label-free research. Document the dataset requirements and the exact commands
that will complete evaluation once data is available. Keep behavioural candidates
separate from the incumbent and state that real-match quality is unverified.
Do not replace the missing evidence with tuning on the public reference, and do
not declare the objective achieved merely because experiments run successfully.

## 9. Freeze, reproduce and prepare the result

Once a method is independently selected, freeze its code/configuration and
generate its competition output locally. Validate the exact final file with the
unchanged official validator. At this stage an allowed public pipeline check may
check alignment, format and label semantics; it must not select the method or
trigger further model tuning. A demonstrated software contract bug needs a
general test-backed correction, not a scene-specific output patch.

The result must contain exactly 199,319 data rows, frames `0..199318`, header
`frame,label`, labels H/A/D/X and no empty cells or extra index column. Respect the
tactical timeline and published boundaries through shared metadata.

Run independent regenerations and compare their bytes. Pin actual dependencies,
record random seeds and model checkpoints, and ensure model/data resources are
accessible to organisers. If adding code outside the current manifest's glob,
extend provenance to cover every inference-relevant source and model asset;
otherwise the present manifest would miss part of the pipeline.

Update `solution/README.md` and `solution/TECHNICAL_REPORT.md` with the final method,
assets, training data, provenance, actual results, limitations and reproduction
command. Update the guide only where behaviour changed. Keep the technical
submission concise; it is not this long masterprompt.

Re-read the current submission requirements before finalising. At this snapshot
they require the CSV, a public repository URL and a short Markdown/PDF technical
document, with a stated deadline of 17 November 2026 and no time of day specified.
The team name must match registration. Prepare artifacts; do not submit them.

## 10. Operating boundaries and first actions

Stay on the user's current development branch, expected `dev_VG`. Do not push to
`testing`, `main` or any other branch. Do not switch, merge, rebase or publish
branches without a separate instruction. Do not commit unless requested. No paid
services, external deployment, messages to others or remote submission are
authorised by this prompt. Routine reversible code changes and local tests within
this task are authorised.

Follow actual tool permissions. Do not bypass a sandbox or approval rejection.
If essential access is blocked, report the concrete action and reason and continue
unaffected work. Ask only when missing information affects legality, required
access or an irreversible decision; do not request confirmation for routine edits.

The interpreter currently used from the repository root is
`.\.venv\Scripts\python.exe`. Verify it rather than assuming `python` is on PATH.
Current safe starting commands are:

```powershell
git status --short --branch
git log -5 --oneline
.\.venv\Scripts\python.exe -m unittest discover -s solution/tests -v
```

After preserving existing outputs, a fresh incumbent reproduction and format check
can be placed in a separate ignored directory:

```powershell
.\.venv\Scripts\python.exe -m solution.code.possession --team MesQueUnCode --output-dir solution/reports/incumbent-recheck
.\.venv\Scripts\python.exe evaluation/validate_submission.py solution/reports/incumbent-recheck/MesQueUnCode.csv
```

Do not automatically re-run public scoring as part of the development loop.

Start now with discovery and a short German assessment of the current method,
constraints, missing independent evidence and the first testable hypothesis.
Then implement the most valuable permitted next step and continue through testing
and review. Use independent code/rule review where it helps, without sending
restricted assets to reviewing assistants.

In the final report, clearly distinguish:

1. What changed and why, with concrete source references.
2. Which tests and experiments actually ran.
3. Improvement or regression on independent data, including uncertainty.
4. Public pipeline observations, explicitly separate from selection evidence.
5. Runtime, reproduction and rule compliance.
6. What is verified, merely implemented, experimental or blocked.
7. Whether the incumbent or a candidate remains the recommended default and the
   legitimate evidence supporting that decision.

The goal is a demonstrably stronger algorithm. If the available data cannot prove
that, deliver the useful engineering work and a precise next step without claiming
an unmeasured victory.
