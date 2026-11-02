# Experiments

What was measured, how to reproduce it, and how far each number can be trusted.

## Three kinds of evaluation, kept apart

| | what | data | used for |
|---|---|---|---|
| **local, unlabelled** | possession count, mean length, stoppages, H/A/D shares over all 158,857 graded frames; silver-label agreement | gated bundle, no labels | designing and sanity-checking features |
| **public validation** | official scorer on the 5-minute public ground truth (17 possessions) | `evaluation/gt/*_public.csv` | checking the pipeline is *correct*; never choosing parameters |
| **final submission** | official scorer on the full-match private ground truth | organisers only | the ranking |

No parameter in `solution/config.py` was chosen by looking at the public columns below.
[EVALUATION.md](EVALUATION.md) shows why: on the public window the ±125 baseline beats the ±50
one, and on the full match it is the other way round.

## Reproduce

```bash
python starter/baseline.py                      # the four reference baselines -> evaluation/results/
python solution/experiments.py                  # the table below
python solution/run_solution.py --output evaluation/results/mes_que_un_code.csv
python evaluation/run_evaluation.py --bootstrap 1000
```

`experiments.py` rebuilds the baselines from the cached tracking with `starter/baseline.py`'s own
smoothing function and asserts that they are identical to the CSVs `starter/baseline.py` wrote. All
four matched exactly, which confirms the vectorised loader and frame alignment.

## Results (measured 2026-11-02, Python 3.14, numpy 2.3.5, pandas 3.0.1)

| system | possessions | mean poss (s) | stoppages | H % | A % | D % | public macroF1 | public spellF1 | public poss | public dur (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline_raw | 1949 | 1.9 | 150 | 33.5 | 25.9 | 40.7 | 0.5972 | 0.0378 | 131 | 2.0 |
| baseline_smooth25 | 746 | 5.4 | 133 | 36.4 | 27.1 | 36.5 | 0.6200 | 0.2868 | 36 | 7.6 |
| baseline_smooth50 | 594 | 7.2 | 133 | 38.6 | 29.1 | 32.3 | 0.6207 | 0.3256 | 26 | 10.8 |
| baseline_smooth125 | 398 | 12.4 | 79 | 44.0 | 33.7 | 22.3 | 0.5812 | 0.4242 | 16 | 18.5 |
| **solution** | **202** | **18.9** | **110** | 37.7 | 22.4 | 39.9 | **0.8344** | **0.4714** | 11 | 23.6 |
| ablate_no_cleanup | 202 | 18.9 | 110 | 37.7 | 22.4 | 39.9 | 0.8344 | 0.4714 | 11 | 23.6 |
| ablate_no_smart_tagging | 207 | 18.7 | 104 | 37.1 | 23.8 | 39.1 | 0.8091 | 0.3360 | 8 | 33.2 |
| ablate_no_team_shape | 197 | 19.5 | 108 | 38.4 | 22.0 | 39.6 | 0.8344 | 0.4714 | 11 | 23.6 |
| ablate_no_dead_model | 139 | 42.8 | 34 | 53.3 | 40.3 | 6.4 | 0.5042 | 0.3333 | 7 | 38.1 |
| ablate_low_hysteresis | 362 | 10.7 | 103 | 37.3 | 23.6 | 39.2 | 0.7903 | 0.4216 | 20 | 13.2 |

Left block: full match, no labels. Right block: public 5 minutes, ground truth 17 possessions of
mean 14.2 s. "Possessions" on the left counts maximal H/A runs inside the graded halves.

Bootstrap intervals from `run_evaluation.py --bootstrap 1000` (5–95%, public window):

| system | spellF1 | interval | macroF1 interval |
|---|---|---|---|
| solution (`mes_que_un_code`) | 0.4714 | 0.257 – 0.671 | 0.613 – 0.884 |
| baseline_smooth125 | 0.4242 | 0.267 – 0.594 | 0.448 – 0.665 |
| baseline_smooth50 (published floor) | 0.3256 | 0.214 – 0.447 | 0.460 – 0.742 |

Published full-match spellF1 for the baselines (EVALUATION.md): raw 0.0486, smooth25 0.1996,
**smooth50 0.2433 (the floor)**, smooth125 0.2242. Full-match spellF1 for our solution requires the
private ground truth: **run locally with the supplied data** is all anyone outside the organisers can
do, and we make no claim about it.

## What the numbers say — and do not say

* **The pipeline is correct.** Frames are aligned (the rebuilt baselines match byte for byte), the
  file validates, the second half is D until its kick-off, and on the public window the solution
  labels frames far better than any baseline (macroF1 0.83 vs 0.58–0.62).
* **The spellF1 gain on the public window is inside the noise.** The intervals overlap heavily; 17
  possessions cannot separate systems. We do not claim a spellF1 improvement from these numbers.
* **The unlabelled full-match statistics move in the right direction.** 202 possessions instead of
  594, a dead-ball share of 39.9% (football's ball-out-of-play time is typically around 40% of the
  match), 110 stoppages — in line with the set pieces, fouls and throw-ins of a real match.
* **Ablations** (read them as directions, not significance):
  * the dead-ball model is essential — without it D falls to 6% and possessions run 43 s;
  * the Smart Tagging priors matter on the public window (spellF1 0.47 → 0.34 without them) —
    mostly through the set-piece dead-ball windows;
  * team shape changes 5 possessions on the full match and nothing in the public window, as
    expected from a prior that only acts where the ball is invisible;
  * clean-up changes nothing at the default costs: the decoder already leaves no sub-second spells.
    It stays as a guard for lower `switch_cost` settings;
  * lower hysteresis produces more, shorter possessions (362 at 10.7 s).
* **Known bias.** Both the full-match count and the public window suggest we under-segment (11
  predicted against 17 true). See *Known limitations* in [MODEL.md](MODEL.md) for why we did not tune
  it away.

## Self-supervised checks (no ground truth)

| check | result |
|---|---|
| Smart Tagging `Team` vs nearest player within 1.5 m of a visible ball | BUILD UP 0.87, PROGRESSION 0.83, ATTACKING TRANSITION 0.71; DEFENDING IN MIDDLE THIRD 0.17 (mirror) |
| team-shape logistic regression, cross-half on silver labels | 84.4% / 84.0% vs majority 61.1% / 62.7% |
| motion likelihood ratio, ball-out vs ball-at-pass-speed | crosses zero between 1.5 and 1.75 m/s |
| kick-off detection vs published kick-offs | 11189 vs 11185 (1st half), 113722 vs 113742 (2nd half) |
