# MesQueUnCode: technical report

More than a Hack 2027, Phase 1. Current method and verification as of 29 October 2026.

## Method

The system assigns H (home possession), A (away possession), D (dead ball) or
X (outside the analysed halves) to every tactical-video frame. It uses supplied
automatic tactical tracking and Smart Tagging; inference requires no human
labels, network service, GPU or model download.

The loader streams tracking in bounded blocks, checks consecutive source frames
and maps one-based ATD indexing to zero-based CSV frames. Distances use pitch
metres. Track slots identify team membership, not stable player identities.

Team evidence combines the nearest-player distance difference with bounded
soft tactical support. Geometry uses a 2.5 m distance scale, a 6 m distance decay
and reduced support for fast loose balls. Attacking tags support their named
team; defensive tags support its opponent. Correlated tags combine by maximum,
with team strength 1.25; they never directly supply output labels.

Live/dead evidence combines collective movement, ball movement, observation
availability, stationary-ball/slow-player evidence and outside-pitch positions.
A deterministic two-component movement mixture is fitted without labels; its
faster component is movement evidence, not a learned live/dead class. Automatic
set-piece windows supply soft restart evidence from movement onset. Missing ball
observations are not automatically D; only short, bounded gaps up to 0.32 s are
interpolated for movement calculations.

An exact offline Viterbi decoder selects the state sequence. Internal states
last at least 0.32 s; boundary states may be shorter. Switching to/from D costs
24. The incumbent H/A switch cost is 20; the responsive candidate uses 12.
Weak transient evidence is discouraged from changing the owner, while stronger
sustained evidence can establish control. These are imperfect control proxies,
not a guarantee of distinguishing every deflection or short possession.

Temporal processing resets at each published half; frames outside the halves
receive X. No manually chosen match timestamps or corrections are used.
Broadcast tracking and homography are not included; synchronised video evidence
is not part of the implemented method. Inference is linear in frames for a fixed
number of tracking slots and decoder states.

## Independent development and data provenance

The external dataset is provided by the **Deutsche Fussball Liga (DFL)** under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): Bassek, Rein, Weber and
Memmert (2025), [An integrated dataset of spatiotemporal and event data in elite soccer](https://doi.org/10.1038/s41597-025-04505-y),
Scientific Data 12, 195. Version used:
[10.6084/m9.figshare.28196177.v1](https://doi.org/10.6084/m9.figshare.28196177.v1).

Native provider BallStatus/BallPossession fields supply the reference, attributed
to DFL/Sportec Solutions and distributed by the authors. Status 0 maps to D;
status 1 with possession 1/2 maps to H/A; other values remain X. These are provider
control/stoppage references, not new manual annotations under hackathon guidelines.
Features use observed positions, speeds and metadata only. No reference-derived
tactical tags, label-dependent observation masks or hand corrections are used.

J03WMX/J03WOH supplied development data (283,181 source frames); J03WN1/J03WOY
were reserved for final evaluation (284,097 frames). All are native 25 Hz.
No supervised model was fitted. These other-match labels were used to select a
switch cost, never to construct inference inputs. Competition public labels were
not used for training, tuning, feature selection, promotion or rollback.

The predeclared [control-cost protocol](experiments/control-switch-cost-002.json)
compared costs 4/8/12 with incumbent 20. Cost 4 failed primary/secondary gates;
cost 8 failed a per-match macroF1 regression limit. Cost 12 alone passed and was
frozen before one final evaluation. Per-match primary, macroF1, D recall,
IoU-0.5 recall, transition and runtime gates passed on both final games.

| Split and match | Incumbent spellF1 | Responsive, cost 12 |
|---|---:|---:|
| Development J03WMX | 0.374570 | 0.390000 |
| Development J03WOH | 0.425806 | 0.445630 |
| Development mean | 0.400188 | 0.417815 |
| Final J03WN1 | 0.326418 | 0.377070 |
| Final J03WOY | 0.383003 | 0.402102 |
| Final mean | 0.354710 | 0.389586 |

The final primary gain is +0.034876 absolute (+9.83% relative). At IoU 0.5,
matched possessions total 252 -> 305, while unmatched predictions rise
221 -> 285. Final-match macroF1 changes are +0.000758/-0.002357. This is a
recall/precision trade-off; two held-out games do not establish significance or
private competition quality. J03WOH and J03WOY share Fortuna Duesseldorf, and
multicamera tracking is cleaner than competition ATD. The split is match-disjoint,
not team- or domain-disjoint. Both final games are now consumed and excluded
from further design or selection.

The [reliability experiment](experiments/motion-reliability-001.json) rejected
isolated motion-inconsistent ball observations using the existing 45 m/s bound.
Its J03WMX pilot predictions were identical to incumbent; no natural-match gain
was established. It remains an optional experiment.

The responsive convenience default was set to the already selected cost 12 after
final evaluation. Frozen prediction hashes were reproduced after that packaging
change without rereading final reference labels or recomputing scores. Incumbent
remains the CLI default for compatibility. New development should beat responsive
12 on independent data. Adapter and evaluation commands are in
[INDEPENDENT_VALIDATION.md](INDEPENDENT_VALIDATION.md).

## Public pipeline check and runtime

This check used fixed methods, 7,500 public frames and 17 reference possessions.
It was not a selection experiment. The baseline uses tracking; our methods also
use the supplied automatic tactical tags.

| Public metric | Original baseline_smooth50 | Incumbent | Responsive 12 |
|---|---:|---:|---:|
| Mean spellF1 | 0.3256 | 0.4444 | 0.4138 |
| MacroF1 | 0.6207 | 0.7362 | 0.7278 |
| Matched phases at IoU 0.5 | 7 | 7 | 7 |
| Unmatched predictions at IoU 0.5 | 19 | 3 | 5 |

For responsive versus baseline, three measured full CLI runs after one warm-up
on 29 October gave medians 10.6895 s versus 4.5314 s (2.36 times as long).
Timing includes startup, reading, inference and output; our solution additionally
validates output and writes provenance. It is not an isolated decoder benchmark.

Public spellF1 5th?95th bootstrap ranges (1,000 draws) were 0.2207?0.6076 for
responsive and 0.2140?0.4465 for baseline. They are not a paired test of the gain
or estimates of private full-match performance. No parameter or selection changed
in response to this check, including the incumbent's higher public score.

## Verification and reproduction

All 91 tests passed on the recorded implementation, including exhaustive small
Viterbi problems, input alignment, short control versus deflections, half resets,
label/input separation, diagnostics invariance and evaluation protocol guards.
`pip check` found no broken requirements; no extra runtime dependency was added.
The pinned environment was already installed locally; a fresh online dependency
installation was not tested.

Both profiles regenerate valid, byte-identical 199,319-row CSVs. X occurs only
outside published halves. SHA-256:

- Incumbent: `3ce21a4bf873fcfe7a04013408537d230e8a32d361ea18ec430d6e12c77bd43e`.
- Responsive 12: `368f50cd5d6d826b97c80e1d492dd474392429ed1430ce60ead243c62d1ebea2`.

From the repository root, with data in the official layout:

```powershell
.\.venv\Scripts\python.exe -m pip install -r solution/requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s solution/tests -v
.\.venv\Scripts\python.exe -m solution.code.possession --profile responsive --team-switch-cost 12 --team MesQueUnCode --output-dir solution/reports/responsive-12
.\.venv\Scripts\python.exe evaluation/validate_submission.py solution/reports/responsive-12/MesQueUnCode.csv
```

Use the registered team name and `--data-dir BUNDLE_ROOT` when needed. Outputs
include the CSV, source/input/configuration manifest and this technical document.
Restricted assets and frame traces remain local and ignored by Git.

## Limitations

Distance is an incomplete control proxy. Long occlusions, noisy tags, short
possessions and interruptions while players keep moving remain difficult.
Movement regimes and evidence scores are not calibrated possession probabilities.
Perfect recognition, private-score improvement and competition victory have not
been demonstrated. Further development requires new held-out matches; the public
competition window and the consumed final games must not become tuning sets.
