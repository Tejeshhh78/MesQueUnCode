# Masterprompt: improve possession control in staged experiments

Execute this task when asked to start the next development round. Creating or
editing this document alone does not start experiments. Explain work in German;
keep code, comments, CLI output and repository documentation in English.

## Objective and scope

Develop a possession detector that improves complete-phase recovery over the
current **responsive profile with H/A switch cost 12**. Preserve reliable team
assignment and stoppage detection. Perfect recovery is an aspiration, not a
promise: incomplete observations can leave control or interruptions ambiguous.

Proceed through the stages below. Implement and verify the useful next step;
do not stop at a plan. Later mechanisms are conditional options, not a checklist
of features to build. Keep the simplest method supported by independent evidence.
Do not choose a model because its public competition score looks attractive.

## Read and establish the baseline

Start with Git status, current branch/history and any applicable AGENTS.md.
Preserve existing edits. Work under `solution/` on the current development branch;
do not switch branches, commit, push, publish, deploy or submit without a new
instruction. No paid services or messages to others are authorised.

Read the official [challenge](../docs/CHALLENGE.md), [rules](../docs/RULES.md),
[FAQ](../docs/FAQ.md), [evaluation](../docs/EVALUATION.md),
[data](../docs/DATA.md) and [submission](../docs/SUBMISSION.md) instructions.
Then inspect:

- [README](README.md), [method and results](TECHNICAL_REPORT.md),
  [algorithm explanation](ALGORITHM_GUIDE.md), [evaluation workflow](INDEPENDENT_VALIDATION.md).
- `code/possession/data.py`, `model.py`, `diagnostics.py` and `__main__.py`.
- `code/independent_evaluation.py`, `import_idsse.py`, `control_experiment.py`.
- `tests/`, `experiments/`, pinned dependencies and the unchanged official scorer.

Verify this starting context rather than assuming it is still current:

- `incumbent` remains the CLI default, with H/A cost 20.
- `responsive` uses cost 12 and is the primary baseline for NEW development.
- `reliable` is a separate ball-position filter without demonstrated natural-match
  improvement. Do not combine it with another change without an ablation.
- The original hackathon `baseline_smooth50` is a secondary reference. Beating it
  alone is insufficient to improve our selected method.
- Features include ball positions, nearest-team distances, nearest-player scalar
  speeds, mean speed, fast-player fraction and visible-player count. Of these,
  `player_speed`, `fast_fraction` and `player_count` are extracted but not used
  by the current emissions. Verify this in code.
- Evidence is label-free at inference, including per-match unsupervised movement
  fitting. Viterbi already uses offline future context and resets per half.
- Track slots and changes in the nearest player do not establish stable identities.

Run the existing tests and snapshot the responsive source/configuration and its
output hash before behaviour changes. The recorded competition CSV SHA-256 is
`368f50cd5d6d826b97c80e1d492dd474392429ed1430ce60ead243c62d1ebea2`.
Explain any reproduction discrepancy before proceeding. Do not run public scoring
as part of baseline establishment; format validation needs no reference labels.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s solution/tests -v
.\.venv\Scripts\python.exe -m solution.code.possession --profile responsive --team-switch-cost 12 --output-dir solution/reports/control-evidence-baseline
.\.venv\Scripts\python.exe evaluation/validate_submission.py solution/reports/control-evidence-baseline/MesQueUnCode.csv
```

Use a fresh ignored output directory if the example already exists. Give a brief
German assessment of confirmed facts, uncertainties and the first hypothesis,
then continue autonomously.

## Boundaries that apply to every stage

1. No training, tuning, feature/architecture selection, promotion or rollback
   based on the competition's public five minutes, private labels, or their
   derived counts, durations, errors or scores. Splitting/bootstrapping the public
   window does not make it a permissible development set.
2. No human annotation/correction of the competition game or external human-made
   events, corrected tracking or possession labels about that same game.
   Automatic unlabelled adaptation is permitted under the actual rules.
3. Other-match data must have inspectable sources, licences, label provenance
   and organiser access. Proxy labels remain proxy evidence; agreement with our
   predictions or automatic tags is not independent accuracy.
4. Do not transmit restricted match assets, frame labels, event excerpts, traces
   or screenshots to hosted LLMs. Process locally and return safe aggregates.
   Do not publish reconstructable restricted data. Keep outputs ignored by Git.
5. Leave official evaluation code/reference files unchanged. Preserve native
   timing and units; unknown reference labels must not be fabricated or filled.
6. Never overwrite prior experiments, reset consumption markers or relabel a
   previously inspected final set as unseen. An actual permission/licence blocker
   stops the affected action, not unrelated safe work.

## Stage 1: reserve legitimate evidence before development

Inventory available data and prior use without examining held-out labels.
Recorded IDSSE roles are:

| Matches | Status |
|---|---|
| J03WMX, J03WOH | Development; permissible diagnostic data |
| J03WN1, J03WOY | Consumed final tests; historical results only, excluded from further design/selection |
| J03WPY, J03WQQ, J03WR9 | Outside the recorded four-match experiment; verify prior use before assigning roles |

Reserve new final games before looking at candidate scores. Group complete
physical matches, including repeated views/crops, in a single split. If supervised
learning is attempted, declare a separate training split and fit preprocessing,
calibration and early stopping only on their permitted splits; never split random
frames of one game across training/development/final evaluation.

Research additional public data when necessary. Verify controlled-possession and
stoppage semantics, dense label coverage, native fps, physical units, licence and
access before importing. Sparse events or automatic possession estimates must not
be silently treated as dense gold H/A/D labels. The same provider or repeated
teams limit generalisation even when match IDs differ.

Prefer observations resembling the noisy competition tracking. Predetermined
missingness/noise stress tests may supplement real data; their masks must depend
only on observations or a declared random seed, never on target labels. Report
natural-match quality separately from synthetic-corruption robustness. Do not
manufacture tactical tags from the external reference.

**Acceptance:** a documented split with provenance and at least two new held-out
matches before any promotion. If this is unavailable, complete diagnostics,
software tests and development experiments, but keep the candidate experimental.
Do not reuse the consumed final matches to fill the gap.

## Stage 2: explain errors on development games

Extend the existing evaluation/diagnostic tools only where necessary. Compare
frozen responsive 12 with the development reference using official matching.
Keep raw timelines local; expose useful aggregate counts and measurements for:

- Wrong-team overlaps, unmatched predictions and missed reference phases.
- Likely false splits and merges, with explicit diagnostic overlap definitions.
- Missed/false stoppages and restart boundary errors.
- Start/end errors, phase-duration groups and observed/missing-ball conditions.
- Geometry/tag disagreement and evidence suppressed or selected by decoding.

Define diagnostic categories before counting them. Overlap patterns are not
proof of a causal mechanism; categories can overlap and need not sum to FN/FP.
Distinguish incorrect evidence from decoder suppression. Do not manually patch
individual matches or optimise toward their phase counts.

**Acceptance:** identify one reproducible failure on permissible development
observations, formulate a physical/temporal explanation and create independently
invented synthetic counterexamples. Select the next mechanism using these data
and code evidence, not the public competition results.

## Stage 3: test temporal control evidence first

Default first hypothesis: nearest-player distance sometimes mistakes proximity
for control. Evaluate whether short-term persistence and credible movement
agreement can distinguish a genuine controlled reception from a nearby opponent,
passing ball, deflection or aerial/loose-ball passage.

Reuse existing features when sufficient. Extend extraction only if necessary;
scalar speeds are not relative velocity vectors. A changing nearest player must
not create a fictitious player trajectory. Validate any short-term association
before using direction, acceleration or identity-dependent continuity.

Preserve possession during loose/contested play until control changes according
to the challenge definition. Internal uncertainty need not become a new output
label. Missing ball data must not become an automatic stoppage or turnover.

Add one opt-in candidate with bounded evidence; keep responsive 12 runnable.
Initially hold movement/stoppage evidence, tags and decoder costs fixed to isolate
this change. Predeclare its parameters and maximum number of variants before
candidate evaluation. Do not sweep many arbitrary durations until a score rises.

Test competing cases: strong short control versus brief contact; weak persistent
proximity; fast passes; stationary live ball versus stoppage; missing observations
with and without identifiable ownership change; team swaps, pitch reflection,
half resets and deterministic ties. State when a scenario is unidentifiable.

**Acceptance:** tests pass, feature/label channels stay separate, and the isolated
change improves the predeclared development objective without unacceptable
regressions. Otherwise retain the negative result and reject or revise the
hypothesis on development only. Do not open final data to decide what to try next.

## Stage 4: conditional reliability and transition experiments

Proceed only if Stage 2/3 evidence justifies the added mechanism:

- Weight evidence by observation completeness and consistency. Visible counts,
  player-speed distributions and tag disagreement are possible inputs, not
  automatically useful features. Avoid bias against a partly missing team.
- Test evidence-dependent H/A transition penalties: react to credible control,
  resist unsupported flicker. Explicitly test delayed real changes after occlusion.
- Keep D transitions separate in the experiment unless a development diagnosis
  specifically supports changing stoppage/restart modelling.

Change one component at a time. If the decoder gains frame-dependent costs or
explicit durations, extend the exhaustive small-problem oracle and preserve
half resets, boundary behaviour and deterministic ties. Compare component
ablations so a combination cannot conceal which change helped.

## Stage 5: optional learned evidence

Attempt this only when sufficient licensed training/development data and a
specific diagnostic need justify it. Start with a small interpretable classifier
or tabular model against hand-weighted evidence; retain temporal decoding.
Separate live/dead and control evidence where useful. Do not add a large sequence
model or video pipeline for prestige.

Require reproducible training, fixed seeds, split-aware fitting, inspectable
weights/assets, missing-observation checks and honest calibration assessment.
Native handling of missing values does not by itself prove robustness to the
competition's missingness. Verify generalisation across matches/providers where
possible. Video is optional only if available, licensed and automatically aligned;
never create human corrections for the competition match.

## Stage 6: freeze a candidate and run one new final test

Before any candidate scores, store one compact protocol under `experiments/`
with an unused ID: hypothesis, exact change, dataset roles, fixed baseline,
variant budget, metrics, regression limits, stopping rule and data/source hashes.
Keep completed protocols 001/002 and `control_experiment.py` intact. Extend or
reuse generic evaluation infrastructure rather than its already-consumed search.

Default engineering gates for this new round (not official rules or a statistical
significance guarantee) are:

- At least two development and two genuinely new final matches.
- Equally weighted mean official `spell_f1_mean` gain >= 0.02 over responsive 12.
- No per-match primary loss; per-match macroF1 loss <= 0.005.
- Per-match D recall, IoU-0.5 recall and transition-F1 at 12 frames losses <= 0.02.
- Report IoU-0.5 precision, FP/FN and all official IoUs; reject a method whose
  apparent benefit comes from an undeclared metric or evidence-scope change.
- Median prediction runtime <= 1.5 times the fixed baseline, using at least three
  alternating measured runs after warm-up. Also report whole-CLI timing separately.

If another gate is scientifically justified, declare the change and rationale
before candidate results; never relax it after a failure. Keep variant search
small and explicitly bounded. Rank only eligible candidates; predeclare ties.
Retain the baseline when none qualifies and leave final data unopened.

Freeze the selected source/configuration/assets, then evaluate only that candidate
and responsive 12 once on the new final matches. Report per-match paired changes,
uncertainty appropriate to games and all failed secondary gates. Do not treat
correlated frames as independent samples or claim significance from separate
bootstrap intervals. A final failure ends that experiment; do not retune on it.

## Stage 7: reproduce, document and stop

For an independently supported candidate, regenerate competition predictions
locally and validate the exact 199,319-row CSV. Regenerate independently and
compare bytes. An implementation-only change must preserve existing predictions;
a purposeful model change needs its own hash. Keep the prior model executable.
Do not run a public-score selection loop or interpret a public decrease as a
reason to roll back an independently chosen method.

Record every inference-relevant source/model asset, environment, configuration,
seed and input/output hash. Extend provenance when adding files outside the
current package glob. No manual postprocessing may be necessary for regeneration.

Keep documentation minimal:

- README: installation and actual run/test commands.
- TECHNICAL_REPORT: current method, data/licence, verified results and limitations.
- INDEPENDENT_VALIDATION: dataset roles/contracts and evaluation commands, only if changed.
- ALGORITHM_GUIDE: update explanations only where algorithm behaviour changed.
- One protocol and compact machine-readable results per experiment; detailed
  artifacts remain ignored. Preserve failed/neutral outcomes without a long diary.

Do not add another masterprompt, WORKLOG, progress diary, presentation or redundant
report catalogue. Do not append statements about which prompt or assistant was
used. Preserve official documentation, required submission artifacts and existing
local evidence. Ask only for a genuine missing authorisation or critical rule
ambiguity; otherwise make reversible decisions and continue.

Finish with a concise German account of the implemented change, actual tests,
independent paired gain/regression, runtime, limitations and recommended profile.
Clearly distinguish verified software behaviour, measured independent quality
and unproven transfer to the competition match. If progress stops, leave the
specific next action and missing evidence rather than claiming completion.
