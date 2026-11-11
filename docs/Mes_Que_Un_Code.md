# Mes Que Un Code — technical document

**More than a Hack 2027 · Phase 1: Ball Possession**

Team **Mes Que Un Code**: Tejesh Simkhada, Valentin Gehrig, Leo Hofmann, Mike Gisler, Maxim Krüger.
Repository: <https://github.com/Tejeshhh78/MesQueUnCode>, branch `dev-TS/Setup`.
Submission file: `Mes_Que_Un_Code.csv` (199,319 rows, SHA-256
`8c85d42627fb1af55d444dc61708ea12544e2a49dc831701d126186c2673d088`).

---

## 1. Approach

The ranking metric, spellF1, scores each possession as one interval matched by temporal IoU. One
wrong frame inside a long possession splits it in two: the true spell loses its match and a false
spell appears. So we did not build a per-frame classifier; we built a **segmenter**.

1. **Per-frame evidence.** For every video frame, two log-odds scores: *which team could be playing
   the ball* (`team`, + favours home) and *whether the ball is dead* (`dead`, + favours D). Zero means
   "no information", which is the honest value on the 41% of graded frames with no ball.
2. **Three-state Viterbi decoder (H, A, D).** Per half, it finds the label sequence that maximises
   the summed evidence minus a fixed cost for every change of state: 1.0 nats for a hand-over, 3.0
   for the ball going dead, 3.0 for play restarting. A hand-over happens only when the new team's
   accumulated evidence pays for it, and it is placed where the evidence turns, with no lag. Zero
   evidence carries the current state — the challenge's rule that a loose ball stays with the last
   team.
3. **Clean-up.** Possession spells under 1 s are absorbed by their neighbours; dead-ball blips under
   1 s inside one team's possession are bridged (`H D H` would otherwise count as two possessions).
4. **Output.** `H`/`A`/`D` on every graded frame, `X` on ungraded frames (before kick-off, halftime,
   after the whistle). The file is validated with the official checks before the run reports success.

**Why this and not something else.** A decoder with explicit change costs optimises the thing the
metric counts, and every parameter has a physical meaning, so the handful that need tuning can be
tuned on other matches and carried over.

## 2. Assets used, and dropped

| asset | used | how |
|---|---|---|
| Tactical ATD (tracking) | **yes**, as input | ball and player positions in metres; nearest player per team; ball speed; player motion; team shape |
| Smart Tagging | **yes**, as input | weak team prior from possession phases; dead-ball prior at set-piece starts; kick-off windows |
| Tactical homography | **yes, lightly**, as input | where reprojection error > 5 px, team evidence is halved |
| ATD metadata | **yes**, as input | team of each track slot; home team's attacking direction per half |
| `games.py` | **yes** | graded half boundaries (`half_video_blocks`) and asset paths |
| Metrica Sports public sample games | **yes, external** | the only data anything was fitted or tuned on — see section 3 |
| Broadcast video / ATD / homography | **dropped** | own 164,488-frame timeline, cuts and replays, no mapping to tactical |
| Tactical video | **not used** | not as input, not for checking outputs |

**Frame alignment.** Submission row *i* is tactical video frame *i*. ATD `frame_id` runs 1…199,319,
so `video = frame_id − 1`. Smart Tagging is in seconds, so `video = round(seconds × 25)`. The
per-half IdFrame offsets in `games.py` belong to the hidden counter the ground truth was derived from
and are never added to the tracking. Rebuilding the four reference baselines from our loader
reproduces `starter/baseline.py`'s files byte for byte.

## 3. Training, fitting and tuning data

Following the organisers' rule clarification, **everything that is fitted or tuned uses external
data only.** Nothing is trained, fitted, calibrated or tuned on the provided match, its video or its
ground truth.

**External data:** Metrica Sports' three public sample games
(<https://github.com/metrica-sports/sample-data>, © Metrica Sports, shared for the football
analytics community with acknowledgement). They are other, anonymised matches with tracking and
event data.

* **Labels** for those games are derived mechanically from each game's own event data (owner = team
  of the latest on-ball action; dead from ball-out, foul, offside or goal to the next restart). No
  person labelled anything.
* **Tracking** is degraded towards the competition's automatic tracking (ball missing on about 35%
  of live frames, bursty and more often in crowds, positional noise, player dropouts).
* The loader, labeller and degradation are Leo Hofmann's validation harness
  (`solution/external/`).

| what | fitted / tuned how | held-out check (game not used for fitting) |
|---|---|---|
| dead-ball motion curve: centre 1.367 m/s, scale 0.601 | logistic regression of dead ball on smoothed mean player speed (`calibrate.py`) | 82–86% correct vs majority 57–69% |
| team-shape model: 7 weights + standardisation | logistic regression of home possession on team width, length, territory and forward movement (`calibrate.py`) | 82–83% correct (76–81% when the ball is missing) vs majority 51–53% |
| control radius 1.0 m, hand-over cost 1.0, minimum possession 1.0 s | grid search with the official metric code (`tune_external.py`) | nested leave-one-game-out spellF1 0.540 vs 0.466 for the hand-picked values |

The fitted values are stored in `solution/calibration.json` and the tuned values in
`solution/config.py`. On the provided match they are **only applied**. A unit test,
`test_inference_never_fits_anything`, fails if any fitting routine runs at inference time.

All other parameters are fixed design values from football and physics (pitch geometry, pass and
dribble speeds, line widths), stated in `solution/config.py`. Smart Tagging exists only for the
provided match, so its codes were chosen by their meaning and its weights are small fixed priors;
they were not tuned. The public 5-minute ground truth was used only for automatic correctness checks
of the pipeline.

**No other data:** no hand labels on any game, no event data or corrected tracking of the provided
game, no pretrained models.

## 4. Possession rules implemented

| situation | rule |
|---|---|
| controlled ball | the team whose nearest player is close: Gaussian kernel, σ = 1.0 m, on each team's nearest player |
| loose or contested ball | stays with the last team (zero evidence carries the state) |
| single touch, deflection, aerial duel without control | no change unless the new team accumulates evidence ≥ the hand-over cost, and the possession lasts ≥ 1 s |
| pass in flight past an opponent | evidence × 0.15 above 9 m/s |
| ball out of play | D from the moment it is > 0.75 m past a line, and while unseen afterwards ("last seen out stays out") |
| restart | D until the ball is played (players walking, ball stationary ≥ 1.5 s, set-piece tag start); then possession to whoever controls it |
| kick-off | every frame of a half before its detected kick-off is forced to D; possession then to the kicking team |
| goalkeeper holding the ball | ordinary control by the goalkeeper's team |
| change of team through a stoppage | priced as dead + restart, not as a hand-over |
| possession shorter than 1 s | not a possession: absorbed by its neighbours |
| ball invisible | carry the current state; team shape and Smart Tagging priors may move it |
| dead-ball ownership | never asked for: D |
| ungraded frames | X |

## 5. Results and limitations

The provided match's full ground truth is private, and its public window may not be used to choose
anything, so **the quality evidence is on the external games**, scored with the official metric
code, each game by a model that was not fitted or tuned on it:

| | spellF1 | macroF1 |
|---|---:|---:|
| starter `baseline_smooth50`, same degraded tracking | 0.259 | 0.656 |
| our model, hand-picked costs, external calibration | 0.466 | 0.849 |
| **our model, costs tuned on the other two games** (nested leave-one-game-out) | **0.540** | — |
| our model as submitted (costs tuned on all three games — optimistic) | 0.547 | 0.856 |

The baseline scores 0.24–0.28 on these games against its official full-match 0.2433, which suggests
the degraded external games are a reasonable proxy for the competition data.

**Where it fails, and what we would fix with more time:**

* **Short possessions.** Even tuned, the model finds about 0.66× as many possessions as there are
  on the external games (0.48× before tuning); possessions of a few seconds are the main source of
  misses.
* **Transfer.** Three external games with simulated tracking noise are a proxy for the automatic
  ATD; how well the tuned values carry over to the provided match cannot be measured without its
  private ground truth. Smart Tagging does not exist on the external games, so its weights are
  untested priors.
* **Hand-overs hidden in ball gaps.** A sequence like A→H→A inside one long gap is visible only to
  the weak team-shape prior.
* **Fouls in open play** stop play while the ball stays on the pitch and players slow gradually, so
  the dead-ball cues fire late.
* **Future work:** more external labelled games (for example the DFL/IDSSE open dataset) for
  fitting and a held-out check; a hidden semi-Markov decoder with explicit duration distributions;
  ball re-detection in the tactical video for the missing 41%.

## 6. Reproduction

Requires the gated match bundle, unpacked into `./data` or pointed to by `MTAH_DATA_DIR`.

```bash
git clone https://github.com/Tejeshhh78/MesQueUnCode.git
cd MesQueUnCode
git checkout dev-TS/Setup
python -m venv .venv
.venv/Scripts/activate                         # Windows; on macOS / Linux: source .venv/bin/activate
pip install -r requirements-solution.txt       # numpy 2.3.5, pandas 3.0.1 (Python 3.14.6 used)
python solution/run_solution.py --output solution/results/Mes_Que_Un_Code.csv
python evaluation/validate_submission.py solution/results/Mes_Que_Un_Code.csv
```

The output's SHA-256 should be the one at the top of this document; the pipeline is deterministic.
To reproduce the external fitting and tuning (downloads the public sample games into
`data/external/`, about 180 MB):

```bash
python solution/calibrate.py --download --check    # refits and confirms calibration.json
python solution/evaluate_external.py               # leave-one-game-out scores
python solution/tune_external.py                   # the decoder grid search
```

Tests: `python -m unittest discover -s solution/tests -t .` (50 tests; the real-data tests skip
themselves when the bundle is absent).

Code map: `solution/loaders.py` (alignment and parsing), `solution/features.py` (evidence),
`solution/possession_model.py` (decoder), `solution/postprocess.py` (clean-up and writer),
`solution/config.py` (every parameter, with its origin), `solution/calibrate.py`,
`solution/tune_external.py` and `solution/evaluate_external.py` (external data),
`solution/external/` (Metrica loader, labels and degradation).
