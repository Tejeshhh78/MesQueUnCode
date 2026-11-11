# Experiments

What was measured, on which data, and how far each number can be trusted.

## The rule that shapes everything

The organisers' rule clarification (November 2026): anything trained, fitted, calibrated or tuned
must use **external data only** — never the ground truth possession data, never the provided match
or its video. Automatic checks with the provided data remain allowed.

So the experiments are split:

| | data | used for |
|---|---|---|
| **external** | Metrica Sports' three public sample games, event-derived labels, tracking degraded towards the automatic ATD | fitting, tuning, and every quality number below |
| **provided match** | the gated BAR-ATH bundle | inference only; automatic correctness checks (file valid, frames aligned, deterministic) |
| **public 5-minute ground truth** | `evaluation/gt/*_public.csv` | automatic correctness checks only; nothing chosen from it |

## Head-to-head of the team's models (decides the submission)

All models on **identical** degraded external games (Metrica 1–3, seeds 0 and 1), official metric
code, no Smart Tagging (it does not exist for these games). Models fitted or tuned on these games
are scored on games they were not fitted or tuned on.

| model | g1 | g2 | g3 | mean spellF1 | mean macroF1 | possessions found |
|---|---:|---:|---:|---:|---:|---:|
| controlled, as originally built (per-match regime fit) | 0.5584 | 0.5388 | 0.5632 | 0.5535 | 0.8529 | 0.76× |
| **controlled, compliant: regimes frozen from the other two games, position-based movement** | 0.5557 | 0.5174 | 0.5565 | **0.5432** | 0.8385 | 0.76× |
| Tejesh's Viterbi + shape, nested leave-one-game-out | 0.5671 | 0.5115 | 0.5403 | 0.5396 | 0.8547 | 0.70× |
| v2-shape | 0.5372 | 0.5071 | 0.5406 | 0.5283 | 0.8513 | 0.67× |
| v2 (tuned by its author on these games) | 0.5186 | 0.4994 | 0.5104 | 0.5095 | 0.8428 | 0.65× |
| baseline_smooth50 | | | | 0.2585 | 0.6558 | – |

Speed-scale robustness, same games: with movement taken from a speed field, regimes fitted on one
speed scale and applied to another dropped controlled from 0.54 to 0.4165 (Tejesh's model showed
the same failure: 0.540 → 0.451). With movement computed from positions over one second
(`solution/code/possession/motion.py`), a calibration carried across speed scales scored 0.5420.

**Decision:** submit compliant controlled. It ties with Tejesh's model, needs no tuning on these
games, finds the most possessions, is robust to speed scale by construction, and has independent
evidence on real tracking from Valentin's IDSSE protocol.

## Reproduce

```bash
python solution/calibrate.py --download --check    # fit motion + shape on external games
python solution/evaluate_external.py               # whole pipeline vs baseline, external games
python solution/tune_external.py                   # decoder grid search, nested leave-one-out
```

## Fitted parts, scored on a held-out external game

`calibrate.py`, leave-one-game-out (fit on two games, score the third):

| held-out game | dead-ball motion classifier | majority class | team-shape classifier | when ball missing | majority class |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.850 | 0.574 | 0.821 | 0.771 | 0.521 |
| 2 | 0.824 | 0.607 | 0.815 | 0.763 | 0.531 |
| 3 | 0.859 | 0.691 | 0.834 | 0.811 | 0.505 |

Fitted on all three games (`calibration.json`): dead-ball motion centre 1.367 m/s, scale 0.601;
team-shape weights dominated by the width gap between the teams.

Player speeds for the external games are computed from the clean positions and hidden wherever the
degraded player is missing; differentiating the simulated positional jitter would inflate speeds by
up to 1 m/s and push the threshold up (an earlier run gave 1.93 m/s for that reason).

## Whole pipeline on external games

`evaluate_external.py`, official metric code, two degradation seeds per game:

| game | seed | true poss | baseline_smooth50 spellF1 | ours spellF1 | baseline macroF1 | ours macroF1 | ours poss |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 351 | 0.2663 | 0.5674 | 0.6671 | 0.8465 | 244 |
| 1 | 1 | 351 | 0.2849 | 0.5668 | 0.6682 | 0.8553 | 234 |
| 2 | 0 | 325 | 0.2543 | 0.5218 | 0.6609 | 0.8456 | 207 |
| 2 | 1 | 325 | 0.2371 | 0.5393 | 0.6406 | 0.8380 | 215 |
| 3 | 0 | 384 | 0.2467 | 0.5609 | 0.6475 | 0.8811 | 265 |
| 3 | 1 | 384 | 0.2620 | 0.5234 | 0.6505 | 0.8714 | 245 |
| **mean** | | | **0.2585** | **0.5466** | **0.6558** | **0.8563** | 0.66× truth |

The calibration is leave-one-game-out, but the decoder costs were tuned on all three games, so the
"ours" column is slightly optimistic; the honest figure is the nested one below. The baseline's
0.24–0.28 here sits close to its official full-match 0.2433, which suggests the degraded external
games are a reasonable proxy.

## Decoder tuning

`tune_external.py` grid: control radius 0.6/0.8/1.0/1.5 m × hand-over cost 0.25/0.5/1.0/1.5/2.0 ×
minimum possession 0.7/1.0 s, plus the original hand-picked values (1.5 m, 4.0, 1.0 s).

Nested leave-one-game-out (choose on two games, score the third):

| held-out game | chosen | held-out spellF1 | hand-picked values |
|---:|---|---:|---:|
| 1 | 1.0 m, cost 1.0, 1.0 s | 0.5671 | 0.4647 |
| 2 | 1.0 m, cost 0.5, 1.0 s | 0.5115 | 0.4563 |
| 3 | 1.0 m, cost 1.0, 0.7 s | 0.5403 | 0.4780 |
| **mean** | | **0.5396** | **0.4663** |

Best on all three games, and used in `config.py`: **1.0 m, hand-over cost 1.0, minimum 1.0 s**
(mean spellF1 0.5466; 0.7 s scored 0.5464). Tuning raised the possession count from 0.48× to 0.66×
the truth.

## What these numbers do not say

* They are on three anonymised external matches with *simulated* tracking noise, not on the
  competition match. Transfer to BAR-ATH cannot be measured without the private ground truth.
* Smart Tagging does not exist for the external games, so its priors are untested there.
* Two seeds of the same game are not independent matches; the effective sample is three games.

## Correctness checks on the provided match

`python -m unittest discover -s solution/tests -t .` runs 50 tests, four of them on the provided
bundle. They report only pass or fail: the output has 199,319 rows and passes the official
validator; the second half is D before its kick-off; two runs are identical. `solution/experiments.py`
can be run locally to compare the baselines with the solution on the provided match. Like every
inspection of the provided data, keep its output local; never paste it into an AI service.
