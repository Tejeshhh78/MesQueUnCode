# MesQueUnCode – Technical Report

More than a Hack 2027, Phase 1 · Updated: 29 October 2026

## Objective and approach

For each of the tactical video's 199,319 frames, the solution outputs `H`
(home team), `A` (away team), `D` (ball out of play) or `X` (no answer).
The primary competition metric, `spellF1`, evaluates complete possession
intervals. We therefore model state transitions explicitly rather than merely
smoothing independent frame predictions with a large majority-vote window.

The method combines three complementary signals: ball–player geometry for
observations close to control, automatic tactical tags for team evidence during
occlusions, and collective movement as an indication of stoppages. An exact
Viterbi decoder finds a temporally consistent path through `H`, `A` and `D`.
The algorithm runs entirely locally and deterministically, without an external API.

The nearest-player-plus-majority-vote alternative is simple, but brief touches
can cause false changes and missing ball observations can be mistaken for
stoppages. Copying tactical tags directly would equate phases with possession
intervals. We address both weaknesses with soft evidence and temporal switching
costs. This decision was made before the public pipeline check.

## Assets and training data

The solution uses only the supplied automatic tactical ATD metadata, raw ATD and
tactical Smart Tagging. Track slots are used solely through their team
membership; they are explicitly not treated as stable player identities. The
loader requires consecutive ATD frames starting at 1 and maps them to CSV frames
starting at 0. Coordinates are converted to metres using the official pitch dimensions.

**No human labels**, external datasets, private models or human-produced event
data about this match are used by incumbent inference. Independent development
now additionally uses public IDSSE reference labels from other matches, as
documented in [INDEPENDENT_VALIDATION.md](INDEPENDENT_VALIDATION.md); they are
never inputs to possession inference. There is no supervised training. Only two
movement regimes are estimated by unsupervised EM from the collective speed in
the supplied tracking. The faster component supplies movement evidence and is
not output as a learned ground-truth class. This is not calibration against
possession labels. Synthetic scenarios are used only for software tests.

The initial public pipeline check was performed with the official scorer after
the method had been fixed. A subsequent head-to-head diagnostic regenerated and
scored the same fixed solution alongside the preselected `baseline_smooth50`.
Competition ground truth was never passed to inference or used for training,
parameter selection or selecting among candidate methods. The new independent
development workflow keeps those restrictions and evaluates explicit experiments
on other games, with separate development and reserved final-test matches.

The videos are absent locally. Broadcast tracking and homographies were reviewed
for suitability but were not included in inference: broadcast uses a separate,
unsynchronised timeline, while tactical positions are already expressed in pitch
coordinates. No accuracy gain from these omitted assets is claimed. Raw data and
frame-level intermediate records are not exported or sent to cloud models.

## Evidence and possession rules

**Team control.** For each frame, we find the ball's minimum distance to each
team. The distance difference is scaled by 2.5 m and clipped to ±2.5. Exponential
weighting with a 6 m radius reduces the influence of distant players. Fast loose
balls receive a further reduction: the nearest player is not yet a confirmed
receiver. Automatic attacking phases support the named team; defensive phases
support its opponent. Team-tagging strength is 1.25. Overlapping correlated tags
are combined by their maximum on each side rather than accumulated without a
bound. `N/A` supplies no team assignment. Tags remain evidence, not hard
possession intervals.

**Loose balls, challenges and deflections.** Weak or missing evidence for the
other team is insufficient to trigger a change. The decoder then typically
retains the previous state. Strong, sustained evidence can trigger a change.
Switching costs are 20 between teams and 24 for a transition to or from `D`.
Each internal state lasts at least eight frames (0.32 s); states at the boundary
of a half may be shorter. Optimisation uses future observations without imposing
a uniform smoothing-window delay. No particular number of passes is required.

**Ball gaps.** Only gaps of at most eight frames with observations on both sides
are interpolated linearly for movement signals. Longer gaps are not filled with
invented ball trajectories. A missing ball is weak evidence, never a hard `D`
rule. Tagging and temporal continuity remain available.

**Stoppages and restarts.** Mean player movement, ball movement, a stationary
ball combined with low collective movement, and observed positions outside the
pitch supply game-state evidence. Extreme ball speeds are treated as unreliable.
Within automatically marked set-piece windows, we search for ball movement onset
or, as a fallback, an increase in collective movement. Without a clear onset,
only a weak interval prior remains. The whole set-piece window is not labelled
`D` automatically. Dead-ball ownership is not modelled: stoppages receive `D`.

All temporal windows and decoding are computed separately for each published
half. Outside these official boundaries, we output `X`. There are no rules or
corrections for individually selected match events. Physical scales and
regularisation are transparent modelling assumptions, not hyperparameters
empirically optimised against the public ground truth.

## Architecture, complexity and reproduction

`solution/code/possession/data.py` streams the ATD once and retains ball/team features rather
than full player trajectories. `solution/code/possession/model.py` computes evidence, fits the
movement mixture and decodes states. `solution/code/possession/__main__.py` writes UTF-8 with
LF line endings, validates before atomic replacement, and creates a manifest
containing SHA-256 checksums of the inputs, code and output. Inference requires
neither ground truth nor network access.

For efficiency, at most 4,096 input lines are processed together with NumPy.
Frame and numeric validation remain strict. The decoder uses small Python float
lists and compact predecessor bits to avoid millions of tiny array calls. The
state recurrence, tie-breaking rules and model parameters are preserved; the
full-match CSV is byte-identical to the original implementation. No result cache
is used.

Runtime is linear in frames × track slots; the decoder is linear in frames for a
fixed number of states. Tested with CPython 3.12.14 and the packages pinned in
`solution/requirements.txt`. From the repository root:

```bash
python -m pip install -r solution/requirements.txt
python -m unittest discover -s solution/tests -v
python -m solution.code.possession --team MesQueUnCode --output-dir solution/results
python evaluation/validate_submission.py solution/results/MesQueUnCode.csv
```

Place the data under `data/` using the official layout, or specify
`--data-dir BUNDLE_ROOT`. Outputs are `MesQueUnCode.csv`, a copy of this report as
`MesQueUnCode.md`, and a local manifest. Replace the team name with the registered
name using `--team` if necessary. A fresh package download was not tested; the
pinned versions were already installed locally and were confirmed with
`pip install --no-index`.

## Actual verification and results

- 16 synthetic tests passed: strict frame alignment and input validation,
  team assignment in metres, tag timing, short/long ball gaps, deflections versus
  controlled changes, stoppages and restarts, half resets, correlated tags,
  deterministic movement fitting, and the CSV/metric contract. Additional checks
  cover identical features across block boundaries, rejection of partial numeric
  tokens and deterministic decoder ties.
- For small problems with minimum durations of 1, 2 and 3, the decoder was checked
  against every possible state path; the optimal objective value matches.
- The official ±50 baseline was independently regenerated from local raw data;
  all 199,319 labels match the supplied baseline.
- The complete submission passes the official format check. It contains 40,462
  `X` labels exclusively outside the published half boundaries and a total of 210
  algorithmically predicted possession spells.
- Independent complete runs produce identical CSV bytes.
- First complete run: 29.38 s on the available local runtime. This is a single
  observation, not a standardised hardware benchmark.
- After implementation-only optimisation, the median complete CLI runtime was
  13.45 s (12.45–23.07 s), versus 5.62 s (4.09–8.13 s) for the original
  `baseline_smooth50`, each over three measured runs after a warm-up. Our pipeline
  still takes 2.39 times as long. Before optimisation, the first measurement
  series gave 26.74 s versus 4.88 s; machine load and individual timings vary.
  `solution/code/benchmark_runtime.py` includes startup, reading and writing; our pipeline also
  validates output and creates a manifest. This runtime comparison uses no GT.

Initial public pipeline check on 7,500 frames / 17 possession spells:

| Metric | Result | 5th–95th bootstrap percentile, 1,000 draws |
|---|---:|---:|
| spellF1 | 0.4444 | 0.2370–0.6519 |
| macroF1 | 0.7362 | 0.5707–0.8132 |

These values are **not estimates of the private full-match score**. The wide
interval prevents a reliable claim of superiority from this window. No model
parameters were changed based on the score after this check.

## Independent quality experiment on 28 October 2026

The optional `responsive` profile uses the same evidence and decoder with only
the H/A switch cost reduced from 20 to **12**. Minimum duration and stoppage
penalties are unchanged. A predeclared grid of 4/8/12 was evaluated on two other
development matches; 12 was the only candidate satisfying all per-match
regression limits. It was then frozen and tested once on two reserved games.

The external reference is the **Deutsche Fussball Liga (DFL)** IDSSE release,
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), from
[Bassek, Rein, Weber and Memmert (2025), Scientific Data 12, 195](https://doi.org/10.1038/s41597-025-04505-y).
The [versioned public dataset](https://doi.org/10.6084/m9.figshare.28196177.v1)
contains native provider control and ball-status reference fields. We used
J03WMX/J03WOH for development (283,181 source frames) and J03WN1/J03WOY for final
evaluation (284,097 frames), all at 25 Hz. We did not annotate new labels or
fit a supervised model. Reference fields never enter input construction;
positions and speeds remain available during stoppages. No tactical tags were
manufactured from the reference. Source URLs, hashes, provenance and the exact
label mapping are recorded in [INDEPENDENT_VALIDATION.md](INDEPENDENT_VALIDATION.md).

| Equal-match mean spellF1 | Incumbent | Responsive, cost 12 |
|---|---:|---:|
| Two development matches | 0.400188 | 0.417815 |
| Two previously unused final matches | 0.354710 | 0.389586 |

Both final games improve and pass the declared secondary limits. At IoU 0.5,
matched possessions total 252 versus 305, but unmatched predictions also rise
from 221 to 285. The method gains recall at a precision cost. Two held-out games
do not establish statistical significance or the private competition score.
Some games share Fortuna Duesseldorf; this is a match-disjoint, not team-disjoint
split. Multicamera tracking is substantially cleaner than competition ATD.
Single-pass inference timings are too variable for a speedup claim.

The `reliable` observation-filter experiment produced identical predictions to
the incumbent on its independent development pilot and was not selected.
All **91** current synthetic regression, adapter, diagnostics and protocol tests
pass. The selected configuration's final-game prediction hashes were reproduced
after packaging, without opening reference labels or recomputing final metrics.
The official evaluator and competition reference were not changed or used for
variant selection. Detailed traces stay under ignored local output paths.

The incumbent remains the default for compatibility and as the reproduction
reference. To generate the independently supported candidate explicitly:

```powershell
.\.venv\Scripts\python.exe -m solution.code.possession --profile responsive --team-switch-cost 12 --output-dir solution/reports/responsive-12
.\.venv\Scripts\python.exe evaluation/validate_submission.py solution/reports/responsive-12/MesQueUnCode.csv
```

The candidate is a separate artifact; improved competition quality is unverified.
The final split is now consumed and must not be used for further tuning.

On 29 October, a requested public pipeline check compared this already fixed
candidate with the original `baseline_smooth50`: spellF1 0.4138 versus 0.3256,
macroF1 0.7278 versus 0.6207. At IoU 0.5 both match 7/17 reference possessions,
with 5 versus 19 unmatched predictions. Full CLI medians over three measured
runs after one warm-up were 10.69 versus 4.53 seconds. The candidate prediction
hash matches the independently selected, previously regenerated artifact.
No choice, parameter or default was changed in response to this public check;
these five-minute results do not establish private full-match superiority.

## Limitations and useful next steps

Without referee/video evidence, fouls and restarts with players still moving are
hard to detect. Collective movement regimes do not replace ball-in-play labels.
Long occlusions and incorrect tactical tags can conceal or misattribute
possession changes. Switching costs can merge short genuine possessions; there
is no formal guarantee of distinguishing control from every touch. Team evidence
is not a calibrated probability. Private labels are unavailable, so winning or
exceeding the full-match baseline of 0.2433 has not been demonstrated.

The next methodological improvement should be validated on legally available
**other matches** and their labels, especially for stoppages and short
possessions. Local video evidence is a possible later extension once videos are
available. The public competition window remains excluded from tuning. The
current solution contains no manual corrections.
