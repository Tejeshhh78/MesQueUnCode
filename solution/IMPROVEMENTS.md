# Possession improvements — experiment 006

Branch: `experiment/possession-improvements`. Baseline: `fcacb57` on `final/submission`.
The selected `controlled` profile is preserved. All additions are opt-in experiments.

## Implemented variants

| Profile | Change from controlled |
|---|---|
| `position_motion` | Nearest-player speeds computed from positions over +/-0.08 s, consistently across trackers. |
| `contact` | Bounded support for plausible ball-velocity changes near a player; requires complete original observations and rejects implausible jumps. |
| `gap_memory` | Previously verified possession evidence decays through missing observations instead of becoming zero immediately. |
| `speed_guard` | Unknown ball speed limits instantaneous close-control weight; it is not interpreted as an observed slow ball. |
| `improved` | All four changes together. |

Switch costs, minimum duration, tagging weights and frozen external calibration are unchanged.
No fitting or tuning occurs on the competition match. The original baseline remains the CLI default.

## Results

Evaluated 283,181 frames from the two declared development games with the official scorer.
Both games improve with the combined candidate; baseline predictions exactly match the model
from `fcacb57` on identical inputs.

| Profile | Mean spellF1 | Mean macroF1 | Development gates |
|---|---:|---:|---|
| controlled | 0.510049 | 0.874355 | baseline |
| position_motion | 0.516092 | 0.877301 | pass |
| contact | 0.527677 | 0.877863 | pass |
| gap_memory | 0.510049 | 0.874355 | no clean-data gain |
| speed_guard | 0.508636 | 0.873988 | fail |
| improved | **0.532100** | **0.880024** | **pass** |

The combined gain is +0.022051 spellF1 (+4.32% relative). Per-game spellF1:
J03WMX 0.496569 -> 0.521886; J03WOH 0.523529 -> 0.542313.
At IoU 0.5, pooled matched possessions increase 391 -> 407, unmatched predictions
decrease 192 -> 190, and missed reference possessions decrease 522 -> 506.
Short-phase matches (<2 seconds) increase 23 -> 30 of 301. These remain a major weakness.
Dead-ball recall decreases by less than 0.0001 on either game. Median predict-only
runtime rises by about 5% on each game; this is not a complete CLI benchmark.

On the separately reported artificial ball-gap test, mean spellF1 is
0.475526 -> 0.494827 (+0.019301); both games improve. Gap memory alone helps in
this stress check (0.485119), but has no effect on the clean observed-ball games.
The speed guard alone regresses on average, both clean and stressed. It remains a
separate experimental switch; the combined candidate was tested exactly as declared,
without removing or retuning components after seeing results. A future experiment
would be needed to determine whether omitting that component improves the combination.

**Decision:** `improved` is a promising opt-in candidate, not a replacement declared
superior on unseen matches. Two reused development games do not establish transfer.
The original submitted default is preserved. In particular, these IDSSE scores must not
be compared numerically to the earlier 0.543 score on the different Metrica dataset.

Machine-readable aggregate results and provenance are stored alongside the protocol in
`solution/experiments/possession-improvements-006.results.json`. Full detailed reports stay
local in `solution/reports/improvements-006/results.json`.

Verification: 139 tests passed. Both generated CSVs pass official validation with 199,319 rows.
The controlled file retains the original submission SHA-256
`3e4b381f8e200a10594240ca3e2a528294d48a60c186720699ee019b8fbbff41`.
The improved file is deterministic across two runs, SHA-256
`61270dfbd362eb07c89717dcb9e14fcadd6533f3de793b6d3e9e4d92d34e7191`.

## Evaluation and reproduction

The protocol was recorded before evaluating candidates in
`solution/experiments/possession-improvements-006.json`. Only the existing development games
J03WMX and J03WOH are used. Consumed final tests and the reserved J03WR9 are not evaluated.
These two development games have been used before, so gains are exploratory, not independent
proof of generalisation. Native tracking is cleaner than the competition tracking and has no
Smart Tagging. Report the artificial missing-ball check separately.

The IDSSE importer now optionally preserves player trajectories during import and computes
collective motion using the same position-based function as inference. Differences never cross
period boundaries. Provider nearest-player speed remains available for the baseline, alongside
the position-derived channel for the experimental variant. Reference exports with legacy
provider collective speed are rejected by the frozen-regime evaluator: re-import them using
`--position-motion`. Historical export files are preserved.

Run from the repository root with the original public IDSSE XML files available locally:

```powershell
python -m solution.code.evaluate_improvements --output-dir solution/reports/improvements-006-new
python -m unittest discover -s solution/tests -t .
```

The runner imports only the two development matches, evaluates all six profiles with the
unchanged official scorer, records three alternating timing runs, and checks that `controlled`
has identical predictions to the model loaded from the immutable baseline Git commit on the
same inputs. It also evaluates a separately reported seed-606 ball-deletion stress scenario.
Existing reports are never overwritten. Full local metrics, short-phase diagnostics, hashes
and configurations are written to `results.json`; generated tracking/reference files remain
Git-ignored. Bootstrap intervals are omitted: this experiment does not establish statistical
significance from two reused games.

To generate an experimental CSV without replacing the final submission:

```powershell
python -m solution.code.possession --profile improved --team Mes_Que_Un_Code --output-dir solution/results/experiment-006-improved
python evaluation/validate_submission.py solution/results/experiment-006-improved/Mes_Que_Un_Code.csv
```

Use `--profile contact` or another named profile to test a single change. Every candidate
manifest is marked `experimental`; producing a valid CSV does not promote the candidate.

## Short possessions — experiment 007

The opt-in `short_control` profile builds on `improved`. A transition H -> A or A -> H
can cost 6..12 instead of 12 only when there is a plausible, fully observed local ball action,
nearby-player evidence, a positive emission margin over both alternative states and consistent
nearest-team direction over three frames. Action timing has a +/-0.12 s tolerance; missing
observations do not qualify. D costs and the 0.32 s minimum duration stay unchanged. Viterbi
optimises the resulting frame-dependent costs exactly.

The protocol `solution/experiments/short-control-007.json` fixed one candidate and acceptance
gates before its evaluation. The previous experiment's stopping rule is preserved: this is a
new separately recorded hypothesis, not a threshold sweep of experiment 006. The two existing
development games were reused; no consumed or reserved final games were scored.

| Metric | improved | short_control |
|---|---:|---:|
| Mean spellF1 | 0.532100 | **0.546723** |
| Mean macroF1 | 0.880024 | **0.881831** |
| J03WMX spellF1 | 0.521886 | 0.538717 |
| J03WOH spellF1 | 0.542313 | 0.554728 |
| Short phases matched at IoU .5 | 30 / 301 | **39 / 301** |
| All phases matched at IoU .5 | 407 | 428 |
| Unmatched predictions at IoU .5 | 190 | 199 |
| Missed reference phases at IoU .5 | 506 | 485 |
| Artificial ball-gap mean spellF1 | 0.494827 | 0.501323 |

This is +0.014623 spellF1 (+2.75% relative) over `improved`, and nine additional short-phase
matches (+30% relative). All fixed development gates pass. Improved baseline predictions
exactly match experiment 006 on identical inputs. The extra nine unmatched predictions are
a real tradeoff; most short phases are still missed (39/301 = 13% matched). Dead recall
changes by less than 0.00005 on either game. The artificial ball-gap check also improves
on both games, with short-phase matches increasing from 21 to 27 of 301.

These gains remain exploratory: the same two development games have been used repeatedly,
and neither significance nor transfer to the competition game is established. Timing ran
alongside other local checks, so no reliable speed comparison is claimed. The new profile
is experimental; defaults are preserved. Aggregate results and hashes are stored in
`solution/experiments/short-control-007.results.json` and full local reports in
`solution/reports/short-control-007/results.json`.

```powershell
python -m solution.code.evaluate_short_control --output-dir solution/reports/short-control-007-new
python -m solution.code.possession --profile short_control --team Mes_Que_Un_Code --output-dir solution/results/experiment-007-short-control
python evaluation/validate_submission.py solution/results/experiment-007-short-control/Mes_Que_Un_Code.csv
```

The evaluator reuses experiment 006's position-based development manifest by default; use
`--manifest PATH` after regenerating it elsewhere. No new raw import is needed. Existing
reports are never overwritten. Verification: 143 tests pass, including exhaustive dynamic-cost
decoder optima, static/dynamic equivalence and missing-data/contact safeguards.

Local output correctness check: the generated `short_control` competition CSV has 199,319 rows,
passes official format validation and is byte-identical across two runs. It is also byte-identical
to the previous `improved` competition CSV (SHA-256
`61270dfbd362eb07c89717dcb9e14fcadd6533f3de793b6d3e9e4d92d34e7191`). Thus the new rule changes
development predictions but currently changes no competition submission labels. No competition
reference was scored and no threshold was adjusted in response to this correctness check.
