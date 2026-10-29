# Independent quality development

Updated 28 October 2026. The incumbent remains the default. The `reliable` and
`responsive` profiles are explicit experiments; selection evidence is reported below.
Competition public labels are excluded from this development workflow.

## What changed

- [Model](code/possession/model.py): `--profile reliable` adds one physical
  consistency check. A ball observation is rejected when both adjacent steps
  exceed 45 m/s, but the two surrounding observations connect at a plausible
  speed. The bound is the existing model's speed plausibility bound. Missing
  neighbours do not establish an outlier; plausible fast play is preserved.
  Team evidence, movement fitting, weights and decoding otherwise stay unchanged.
- `--profile responsive` keeps incumbent emissions unchanged and changes only
  the H/A switching cost. Its independently selected default is 12; an explicit
  `--team-switch-cost` overrides this and is recorded in the output manifest.
  Minimum duration (0.32 s) and all live/dead switching costs (24) stay unchanged.
  The consistency filter is not combined with this experiment.
- [Diagnostics](code/possession/diagnostics.py): optional evidence summaries
  expose geometry/tag disagreements, activity, live/dead evidence, observation
  rejection and temporal overrides. These are uncalibrated scores, not accuracy
  estimates. Tracing does not change predictions.
- [Independent evaluator](code/independent_evaluation.py): runs both profiles on
  identical other-match observations and calls the unchanged official scorer.
  It checks explicit units, native 25 Hz, frame coverage, whole-match splits,
  provenance and separate feature/reference channels. Reference, proxy and
  synthetic results are reported separately. It never promotes a profile.
- [IDSSE adapter](code/import_idsse.py): streams native match files into numeric
  feature arrays and a separate reference CSV. Changing only reference fields
  cannot change features, period boundaries or missingness.

No source under `evaluation/`, no competition reference, and no incumbent
parameter was changed. No model was trained on competition labels.

## External data and attribution

The **Deutsche Fussball Liga (DFL)** supplies the public IDSSE dataset under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Credit:
Manuel Bassek, Robert Rein, Hendrik Weber and Daniel Memmert (2025),
[An integrated dataset of spatiotemporal and event data in elite soccer](https://doi.org/10.1038/s41597-025-04505-y),
Scientific Data 12, 195. Dataset version used:
[10.6084/m9.figshare.28196177.v1](https://doi.org/10.6084/m9.figshare.28196177.v1).
The [authors' repository](https://github.com/spoho-datascience/idsse-data)
documents access, DFL authorisation and attribution requirements.

This release contains seven other matches. The initial pilot selected two by
lexicographic match ID before candidate scores were examined:

| Role | Source match ID | Teams |
|---|---|---|
| Development | DFL-MAT-J03WMX | 1. FC Koeln - FC Bayern Muenchen |
| Reserved final test | DFL-MAT-J03WN1 | VfL Bochum 1848 - Bayer 04 Leverkusen |

The provider's `BallPossession` describes the team controlling the ball;
`BallStatus` distinguishes active play from referee interruptions. The adapter
maps status 0 to `D`, and status 1 with possession 1/2 to `H`/`A`. All other
combinations remain `X`. These are native provider reference fields, not new
human annotations or event-filled pseudo-labels. We do not claim that every
flag was manually adjudicated under the hackathon's annotation instructions.

Inputs use only observed positions, player speeds and metadata. Coordinates
are converted from centred metres to normalised pitch coordinates, speed from
km/h to m/s. The adapter excludes referees, preserves dead-ball observations,
uses metadata periods and creates no tactical tags from the reference. It
records source URLs and SHA-256 hashes, its own source hash, label provenance
and coverage. Halves are concatenated for storage; inference resets at each
period. Evaluation inserts an ignored separation between periods without
dropping observed frames or changing within-period timing.

**Limits:** complete multicamera tracking is a different domain from the noisy
competition ATD. The development match has all 22 players and ball coordinates
throughout. Similar label definitions do not prove identical annotation policy.
One development match cannot establish cross-match generalisation. The pilot's
reserved game was converted mechanically without scoring or design use; it was
later evaluated once in the separate four-game experiment described below.

## Experiment 001: motion reliability

Protocol [motion-reliability-001](experiments/motion-reliability-001.json)
was recorded before candidate evaluation.
The sole hypothesis is that isolated impossible positions supply unsupported
team or stoppage evidence. No weight, switch cost or duration sweep is permitted
for this experiment. A synthetic alternating-position failure is an independent
software/robustness check, not evidence of natural-match accuracy.

The primary quantity is the equally weighted mean of official `spell_f1_mean`
over independent matches. Conservative project promotion criteria require at
least two development and two held-out matches, a primary gain of at least
0.01, macroF1 loss no greater than 0.005, D-recall/IoU-0.5-recall/transition-F1
losses no greater than 0.02, and inference runtime no greater than 1.5 times the
incumbent. These are engineering criteria, not official rules or a significance
guarantee. The initial two-game pilot cannot satisfy them. Final-test evaluation
stays locked until independent development selection is complete.

Per-match bootstrap intervals are descriptive; they are not a paired test of a
gain or an estimate of between-match uncertainty. A point estimate on one game
must not be described as proof that the hackathon score has improved.

Actual independent pilot result on J03WMX, 145,967 source frames:

| Metric | Incumbent | Reliable |
|---|---:|---:|
| Mean spellF1 | 0.374570 | 0.374570 |
| Frame macroF1 | 0.856095 | 0.856095 |
| D recall | 0.942572 | 0.942572 |
| Possession recall at IoU 0.5 | 0.273973 | 0.273973 |

Predictions are **exactly identical**, including their hash. The 5th–95th
within-match bootstrap interval for spellF1 was 0.3332–0.4143 (1,000 draws) for
both. Inference-only times were 0.829/0.795 seconds in one run; the small timing
difference is not a speedup claim. The result does not establish a natural-match
quality benefit for the filter. It remains experimental and was not promoted.

The independent pilot also exposes a useful development hypothesis: high
dead-ball recall coexists with low recovery of complete possession phases.
This motivates a separate temporal experiment; it does not justify tuning
towards a particular possession count.

## Experiment 002: control-switch costs

The [predeclared protocol](experiments/control-switch-cost-002.json) compares
exactly three costs, 4, 8 and 12, against incumbent 20, with no other model
change. It was recorded before these candidate scores or additional match
labels were examined. Two more games were selected by the next lexicographic
source IDs, preserving the original reserved test:

| Split | Complete match IDs |
|---|---|
| Development | J03WMX, J03WOH |
| Reserved final test | J03WN1, J03WOY |

J03WOH and J03WOY both feature Fortuna Duesseldorf: this is a match-disjoint,
not team-disjoint split. It cannot establish generalisation to unseen teams or
to the competition's noisier tracking domain.

[control_experiment.py](code/control_experiment.py) selects the largest mean
spellF1 among candidates satisfying the protocol. The mean gain must be at
least 0.01; no match may lose primary score, and secondary regression/runtime
limits apply to **each match**. Exact ties prefer the higher cost. If no variant
qualifies, the final labels are not opened. Otherwise source, manifest,
protocol and configuration hashes freeze the selection for one final evaluation
on both reserved games. The same gates apply there. There is no automatic
promotion or retuning after final results.

The fixed cost grid changes the evidence required for a team switch, not the
definition of ground-truth possession. Synthetic checks cover clear short
control, brief glances, weak proximity, unchanged stoppage penalties and half
resets for all three costs. These checks do not replace real-match evaluation.

Additional original files for the four-game experiment:

| Match | Match information | Observed positions |
|---|---|---|
| J03WOH | [51643478](https://ndownloader.figshare.com/files/51643478) | [51643520](https://ndownloader.figshare.com/files/51643520) |
| J03WOY | [51643481](https://ndownloader.figshare.com/files/51643481) | [51643523](https://ndownloader.figshare.com/files/51643523) |

Download these into the same source directory using original Figshare filenames.
They add approximately 712 MB. All downloaded originals were checked against
published MD5 values; the importer records SHA-256 checksums.

```powershell
.\.venv\Scripts\python.exe -m solution.code.import_idsse --source-dir solution/reports/external/idsse --outdir solution/reports/external/idsse-four-match-v1 --development J03WMX J03WOH --final-test J03WN1 J03WOY
.\.venv\Scripts\python.exe -m solution.code.control_experiment select --protocol solution/experiments/control-switch-cost-002.json --manifest solution/reports/external/idsse-four-match-v1/manifest.json --output-dir solution/reports/quality-development/control-switch-cost-002
```

Final-test execution is conditional on a valid frozen selection. Do not run it
to guide further design. Inspect an existing report instead of consuming the
same held-out split again. The driver preserves its start records across output
directories; these prevent accidental reruns, not deliberate record tampering.

For a fresh, qualified reproduction only, the final command is:

```powershell
.\.venv\Scripts\python.exe -m solution.code.control_experiment final-test --selection solution/reports/quality-development/control-switch-cost-002/selection.json --output-dir solution/reports/quality-development/control-switch-cost-002-final
```

The existing workspace has already consumed that test; read its saved results.

### Executed results and decision

All three development variants were run once. Cost 4 failed primary-score and
secondary regression gates. Cost 8 achieved the best development primary mean
(0.4262), but lost 0.0081 macroF1 on J03WOH, exceeding the predeclared 0.005 limit.
Cost **12** was the only eligible candidate and was frozen before the final test.

| Split | Match | Incumbent spellF1 | Responsive, cost 12 | Paired change |
|---|---|---:|---:|---:|
| Development | J03WMX | 0.374570 | 0.390000 | +0.015430 |
| Development | J03WOH | 0.425806 | 0.445630 | +0.019823 |
| **Development mean** | **2 games** | **0.400188** | **0.417815** | **+0.017627** |
| Final test | J03WN1 | 0.326418 | 0.377070 | +0.050652 |
| Final test | J03WOY | 0.383003 | 0.402102 | +0.019100 |
| **Final-test mean** | **2 games** | **0.354710** | **0.389586** | **+0.034876** |

The final-test improvement is 3.49 score percentage points, or approximately
9.83% relative to the incumbent. Both held-out matches passed all predeclared
gates. At IoU 0.5 the total matched possessions increased from 252 to 305, while
unmatched predictions also increased from 221 to 285. This is a recall/precision
trade-off, not universal error reduction. Final-match macroF1 changes were
+0.000758 and -0.002357; D-recall changes +0.000670 and -0.000634. Transition F1
at 12-frame tolerance improved on both final matches, although it decreased
slightly on development matches within the accepted limit.

Single-pass final prediction times were 1.000/1.467 seconds (incumbent/candidate)
on J03WN1 and 1.541/1.442 seconds on J03WOY. These variable observations pass
the 1.5 runtime ratio gate but do not establish a speed improvement or a stable
overhead estimate. Runtime excludes parsing and export.

**Decision:** responsive with cost 12 is the independently supported candidate.
The original incumbent remains the CLI default and reproducible reference;
the candidate is explicit because transfer to noisy competition tracking is
still unverified. Its profile default was updated from the exploratory cost 8
to the already selected cost 12 after final evaluation. No weight was retuned
on final results; evaluated sources were archived before that default update.
Frozen final records are immutable historical results, not a reason to rerun
the consumed final set after the default update.

Only two held-out games support this estimate. Their paired gains range from
0.0191 to 0.0507; this is a descriptive range, **not a confidence interval**.
The game/track-domain limitations above prevent a statistical superiority claim
for new teams or a prediction of the private competition score. Final labels
are now consumed and must not become another tuning set. Further selection
requires genuinely new held-out matches.

Local records:

- `solution/reports/quality-development/control-switch-cost-002/selection.json`
  and `candidate-4.json`, `candidate-8.json`, `candidate-12.json`.
- `solution/reports/quality-development/control-switch-cost-002-final/final.json`
  and its `evaluated-source/` snapshot.
- Every report contains exact inputs, source, configuration and prediction
  hashes. The complete reference-labelled development and final datasets contain
  283,181 and 284,097 source frames respectively (25 Hz, including stoppages).

Run the selected candidate locally without reading competition ground truth:

```powershell
.\.venv\Scripts\python.exe -m solution.code.possession --profile responsive --team-switch-cost 12 --output-dir solution/reports/responsive-12
.\.venv\Scripts\python.exe evaluation/validate_submission.py solution/reports/responsive-12/MesQueUnCode.csv
```

This generates a separate candidate artifact. It does not replace the incumbent
submission or score either method against the public competition window.
Two full runs (explicit cost and matching profile default) generated the same
199,319-row CSV, accepted by the official format validator. Candidate SHA-256:
`368f50cd5d6d826b97c80e1d492dd474392429ed1430ce60ead243c62d1ebea2`.
The incumbent's original CSV hash remains unchanged. All 90 tests pass; the
environment has no broken requirements and no new runtime dependency was added.

## Reproduce the pilot

Run from the repository root using the existing pinned solution environment.
Download the following four files from the versioned dataset into
`solution/reports/external/idsse/`. Together they occupy approximately 810 MB.
Use their original filenames from Figshare; no login or paid API is needed.

| File | Download | Published MD5 |
|---|---|---|
| J03WMX match information | [51643475](https://ndownloader.figshare.com/files/51643475) | `bc5d5b21dbcf36aa30f4cd0428309f6b` |
| J03WMX observed positions | [51643514](https://ndownloader.figshare.com/files/51643514) | `e4273a2adf5dbd9d4ec42def2f01905c` |
| J03WN1 match information | [51643472](https://ndownloader.figshare.com/files/51643472) | `37b0f85588c0acfb995fe633c7ac2e0d` |
| J03WN1 observed positions | [51643517](https://ndownloader.figshare.com/files/51643517) | `bacb2f609c1d1b0bfc906f6b9d4109f2` |

```powershell
.\.venv\Scripts\python.exe -m solution.code.import_idsse --source-dir solution/reports/external/idsse --outdir solution/reports/external/idsse-pilot-v1
.\.venv\Scripts\python.exe -m solution.code.independent_evaluation solution/reports/external/idsse-pilot-v1/manifest.json --development --bootstrap 1000 --output-dir solution/reports/quality-development/idsse-pilot-v1
```

Both commands preserve existing outputs and require a new directory for a new
experiment. The default importer split is the table above. After an existing
import, run only the second command in a fresh report directory. The development
command does not open the reserved final-test feature or label files. Reports
contain per-profile metrics, paired deltas, inference times, configuration,
environment and source/data/prediction hashes. Timings exclude XML conversion,
feature loading and copying, so they cannot be compared with full CLI benchmarks.

To build another inspectable dataset, use the schema in
[the explicitly synthetic fixture](tests/fixtures/independent_manifest.synthetic.json).
Inline invented arrays are allowed only for synthetic fixtures; real inputs use
ATD, numeric NPZ or JSON feature files. Do not merely rename competition data or
declare it independent. Guards catch common mistakes, not dishonest provenance.

## Diagnostics and safe experiments

```powershell
.\.venv\Scripts\python.exe -m solution.code.possession --diagnostics --output-dir solution/reports/incumbent-diagnostics
.\.venv\Scripts\python.exe -m unittest discover -s solution/tests -v
```

Add `--save-trace` only when local frame arrays are needed. This requires
`--diagnostics` and a repository-local, Git-ignored output directory. Do not
publish or paste competition traces into a hosted assistant. Summaries contain
aggregates only. `--profile reliable` is explicit and defaults to a separate
experimental output directory; it cannot overwrite an incumbent output there.

The competition comparison command `solution.code.compare_algorithms` always
uses the public competition scorer. It is a pipeline check, **not** the tool for
candidate selection. Use the independent evaluator above for development.

Executed experiments and acceptance decisions are recorded in
[WORKLOG.md](WORKLOG.md). Generated source snapshots, inputs and reports remain
under ignored `solution/reports/`.
