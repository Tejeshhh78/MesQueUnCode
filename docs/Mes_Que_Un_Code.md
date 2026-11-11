# Mes Que Un Code — technical document

**More than a Hack 2027 · Phase 1: Ball Possession**

Team **Mes Que Un Code**: Tejesh Simkhada, Valentin Gehrig, Leo Hofmann, Mike Gisler, Maxim Krüger.
Repository: <https://github.com/Tejeshhh78/MesQueUnCode>, branch `final/submission`.
Submission file: `Mes_Que_Un_Code.csv` (199,319 rows, SHA-256 `3e4b381f8e200a10594240ca3e2a528294d48a60c186720699ee019b8fbbff41`).

Submitted model: **controlled** (Valentin Gehrig), made rule-compliant and speed-scale-robust for
the final (section 3).

---

## 1. Approach

spellF1 scores each possession as one interval matched by temporal IoU, so a single wrong frame
inside a long possession splits it in two. We therefore built a **segmenter**: per-frame evidence
for each of the three states, then an exact Viterbi decoder that pays a fixed cost for every change
of state.

**Team evidence** — who controls the ball:

* nearest player of each team to the ball, in pitch metres, with a soft 0.6 m touch-reach margin
  and a 1.8 m Gaussian close-control envelope;
* damping when the ball moves much faster than the nearest player (a pass passing by, not a touch);
* multi-window directional persistence (0.28 s), so a brief proximity does not flip ownership;
* a loose-ball retention state (2.5 s while live, 0.35 s after a stoppage), so a ball in flight keeps
  the last verified team rather than flipping to a nearby opponent;
* Smart Tagging as soft support: attacking phases support their named team, defensive phases the
  opponent, correlated tags combined by maximum. Tags never directly produce output labels.

**Live/dead evidence:** collective player movement (section 3), ball movement, how much of the
ball is observed, a stationary ball with slow players, ball positions outside the pitch, and soft
restart evidence from automatic set-piece windows. A missing ball is never treated as dead by
itself; gaps up to 0.32 s are interpolated for movement only.

**Decoder:** an exact offline Viterbi per half. A hand-over H↔A costs 12, entering or leaving D
costs 24, and internal states last at least 0.32 s. Frames outside the published halves are `X`.

## 2. Assets used, and dropped

| asset | used | how |
|---|---|---|
| Tactical ATD (tracking) | input | ball and player positions; team of each track slot |
| Smart Tagging | input | soft team support and restart evidence |
| `games.py` | input | graded half boundaries, asset paths |
| Metrica Sports public sample games | **external** | fitting the movement regimes; evaluation (section 4) |
| DFL/IDSSE open dataset (CC BY 4.0) | **external** | Valentin's model selection (section 4) |
| Tactical homography | not used by the submitted model | — |
| Broadcast video / ATD / homography | dropped | own timeline with cuts and replays; a camera check recovered none of the tactical ball gaps |
| Tactical video | not used | not as input, not for checking outputs |

Frame alignment: submission row *i* is tactical video frame *i*; ATD `frame_id` − 1 = video frame;
Smart Tagging seconds × 25 = video frame. The per-half IdFrame offsets in `games.py` belong to the
hidden ground-truth counter and are never added to the tracking.

## 3. Training, fitting and tuning — external data only

The organisers' rule clarification: anything trained, fitted, calibrated or tuned must use external
data only — never the ground truth possession data, never the provided match or video.

* **The only fitted part — two movement regimes** (slow and fast collective movement) — is fitted
  by unsupervised EM on **Metrica Sports' three public sample games**
  (<https://github.com/metrica-sports/sample-data>, © Metrica Sports, shared for the analytics
  community with acknowledgement), with tracking degraded towards the automatic ATD. The result
  is committed in `solution/code/possession/activity_calibration.json` and **only applied** on the
  provided match. Valentin's original model fitted these regimes on each input match; for the final
  this was moved to external data.
* **Speed-scale robustness.** Regimes fitted on one tracker's speeds broke when applied to speeds of
  a different scale (measured on external games: spellF1 0.542 → 0.417). Collective movement is
  therefore computed by our own code from player *positions* over a one-second window
  (`motion.py`), identically on the external games and the provided match, instead of the ATD's
  speed field. With that feature, a calibration carried across speed scales scored 0.542.
* **The H↔A cost of 12** was selected by Valentin under a predeclared protocol on the DFL/IDSSE open
  dataset (Bassek, Rein, Weber & Memmert 2025, *Scientific Data* 12, 195; CC BY 4.0), using the
  provider's ball-status and possession fields of those Bundesliga games — other matches.
* All other values are fixed design choices. No parameter was set from the provided match, its
  video or its ground truth; the public 5 minutes were used only for automatic correctness checks.
* Tests fail if anything is fitted at inference time, or if the calibration does not declare
  external provenance (`solution/tests/test_controlled_compliance.py`).

No hand labels on any game; no human-produced data about the provided game; no pretrained models.

## 4. How we chose the model, and how good it is

We compared the team's model lines on **identical** external games — Metrica's three sample games,
two degradation seeds each, labels from each game's own events — with the official metric code.
Each model was scored on games it was not fitted or tuned on wherever it had been fitted or tuned
on these games:

| model | author | spellF1 | macroF1 | possessions found |
|---|---|---:|---:|---:|
| baseline_smooth50 (organisers' starter) | – | 0.259 | 0.656 | – |
| v2 (tuned on these games by its author) | Leo Hofmann | 0.510 | 0.843 | 0.65× |
| v2-shape | Valentin Gehrig | 0.528 | 0.851 | 0.67× |
| Viterbi + team shape, tuned on the other two games | Tejesh Simkhada | 0.540 | 0.855 | 0.70× |
| **controlled, as submitted (regimes from the other two games)** | Valentin Gehrig | **0.543** | 0.839 | **0.76×** |

Controlled and Tejesh's model tie within noise on these games. We submit controlled because it
needs no tuning on these games, finds the most possessions, is robust to speed scale by
construction, and was independently validated by Valentin on real tracking: on two unseen IDSSE
matches its spellF1 was 0.529 against 0.442 for its predecessor (his predeclared protocol, with the
original per-match regime fit; the frozen variant was not re-run on IDSSE).

The baseline scores 0.24–0.28 on the degraded Metrica games against its official full-match 0.2433,
which suggests they are a fair proxy. Our score on the provided match is known only to the
organisers.

**Limitations.** The model still finds only about three quarters of the true possessions;
possessions of a few seconds are the main misses. External games with simulated tracking noise are
a proxy, and Smart Tagging does not exist on them, so the tag weights are untested there. Long ball
gaps hide some hand-overs; fouls in open play stop play while players slow only gradually.

## 5. Possession rules implemented

| situation | rule |
|---|---|
| controlled ball | nearest player within reach, with persistence over 0.28 s |
| loose ball, ball in flight | stays with the last verified team (retention state) |
| touch, deflection, lost aerial duel | no change unless evidence pays the hand-over cost of 12 |
| ball out of play | D, from outside-pitch positions, slow players and a stationary ball |
| restart | D until movement onset; set-piece windows give soft restart evidence |
| missing ball | not dead by itself; current state carried unless other evidence moves it |
| dead-ball ownership | never asked for: D |
| ungraded frames | X |

## 6. Team contributions

* **Valentin Gehrig** — the controlled model, its decoder and evidence, the IDSSE validation
  protocol and experiments, diagnostics, a scan of public challenge forks.
* **Leo Hofmann** — the other-match validation harness (Metrica sample games, event-derived labels,
  ATD-like degradation), used here for fitting and evaluation; profile v2.
* **Tejesh Simkhada** — repository and documentation, an independent model (Viterbi + team shape),
  the rule-compliance work (external calibration, position-based movement), the head-to-head
  comparison, integration, the team website.
* **Mike Gisler** — an early analysis of the baseline's failure modes.

## 7. Reproduction

Requires the gated match bundle in `./data` or `MTAH_DATA_DIR`.

```bash
git clone https://github.com/Tejeshhh78/MesQueUnCode.git
cd MesQueUnCode
git checkout final/submission
python -m venv .venv
.venv/Scripts/activate                 # Windows; macOS / Linux: source .venv/bin/activate
pip install -r requirements-solution.txt
python -m solution.code.possession --team Mes_Que_Un_Code --output-dir solution/results/final
python evaluation/validate_submission.py solution/results/final/Mes_Que_Un_Code.csv
```

The output's SHA-256 should match the one at the top. To reproduce the external fitting and
evaluation (downloads Metrica's public sample games, ~180 MB, into the gitignored `data/external/`):

```bash
python -c "from solution.external import metrica; metrica.download()"   # fetch the games once
python -m solution.code.calibrate_activity --check                       # refit, confirm, score
```

Tests: `python -m unittest discover -s solution/tests -t .`.
