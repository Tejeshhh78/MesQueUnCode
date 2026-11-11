# Team review: who built what, which model is best, and how to finish

Reviewed 2 November 2026 from the remote branches of `Tejeshhh78/MesQueUnCode`, and from the fork
scout report on `research/fork-scout-shape` (commit `e67ae42`). Every model below was run locally
on the provided match with the same data bundle. Nothing was pushed or merged.

## 1. Who did what

| person | branch | commits | contribution |
|---|---|---|---|
| **Valentin Gehrig** | `dev_VG`, `research/fork-scout-shape`, first commits of `dev_LH` | 16 | The team's main model line: the original `possession` pipeline (v1), runtime/quality comparison, algorithm guide, the **IDSSE other-match validation framework** with pre-declared experiments and held-out games, the **controlled** model (current best-validated), 115 tests, the public-fork scout, the `v2-shape` formation cue and a broadcast camera check |
| **Leo Hofmann** | `dev_LH` (merged into `testing`) | 1 large | The **Metrica sample-game validation harness**: event-derived labels, degradation of clean tracking towards the automatic ATD, synthetic tag errors, an oracle error budget, and profile **v2** (beat v1 in 72/72 paired other-match runs) |
| **Tejesh Simkhada** | `dev-TS/Setup` | 6 | Repository setup and structure; an **independent** model (`solution/`: Viterbi over H/A/D, silver-label dead-ball calibration, Smart Tagging priors, self-supervised team shape), 47 tests, the five docs, the technical document, the team website |
| **Mike Gisler** (git author "OnlyNick") | `dev-MG/setup-and-analysis` | 1 | `VERBESSERUNGEN.md`: a sharp analysis of the baseline's failure modes (dead-ball logic, pre-kick-off D, gaps, control radius, tags). **Its window choices were picked on the public 5 minutes, which RULES.md forbids — the ideas are good, the numbers must not be used for selection.** |
| **Maxim Krüger** | – | 0 | No commits on any branch of this repository. Ask whether his work lives elsewhere. |

The `testing` branch is `dev_LH` merged with the early `dev-TS/Setup`; it adds nothing of its own.

## 2. The models, side by side

All were run from their branch on the provided match, unchanged, with outputs written outside the
repo. Each one validates, and each teammate model reproduced the SHA-256 recorded in its report.

| system | full-match possessions | mean poss (s) | D share | public macroF1 | public spellF1 | public poss (truth 17) |
|---|---:|---:|---:|---:|---:|---:|
| baseline_smooth50 (organisers) | 594 | 7.2 | 32.3% | 0.6207 | 0.3256 | 26 |
| v1 / "incumbent" (Valentin) | 210 | 18.6 | 38.6% | 0.7362 | 0.4444 | 10 |
| v2 (Leo) | 244 | 16.0 | 38.5% | 0.7601 | 0.4000 | 14 |
| v2-shape (Valentin) | 243 | 16.1 | 38.4% | 0.7616 | 0.4000 | 14 |
| **controlled** (Valentin, `dev_VG`) | 240 | 16.3 | 38.3% | 0.7546 | 0.3862 | 12 |
| Viterbi + shape (Tejesh, `dev-TS`) | 202 | 18.9 | 39.9% | 0.8344 | 0.4714 | 11 |

**Read the public columns as a pipeline check only.** They cover 17 possessions; the bootstrap
intervals of every system overlap, and choosing between models on them is both against the rules
and, as EVALUATION.md shows for the baselines, likely to pick the wrong one.

Agreement on the 158,857 graded frames (frame agreement / spell agreement, i.e. spellF1 of one
model against the other):

| | controlled | v2 | v2-shape | v1 | Tejesh |
|---|---|---|---|---|---|
| controlled | – | 98% / 0.85 | 97% / 0.83 | 97% / 0.81 | 82% / 0.54 |
| Tejesh | 82% / 0.54 | 83% / 0.59 | 84% / 0.62 | 82% / 0.52 | – |

v1, v2, v2-shape and controlled are one family (96–99% agreement). Tejesh's model is genuinely
different — about one frame in six is labelled differently — so its errors are partly independent
of theirs.

## 3. Whose algorithm is good

The question that decides it is **which model has evidence of quality on matches other than the
provided one**, because that is the only legitimate quality signal.

| model | other-match evidence | verdict |
|---|---|---|
| **controlled** | IDSSE (DFL, CC BY 4.0, provider ball-status/possession labels), predeclared protocol: development games J03WMX/J03WOH, then two **unseen** final games J03WPY/J03WQQ: mean spellF1 **0.529** vs 0.442 for its predecessor (+0.087 on both games), macroF1 +0.03. | **Strongest.** Best-validated, reproducible, well-tested. The right base for the final. |
| v2 / v2-shape | Metrica sample games with simulated ATD degradation: v2 ≈ 0.51, v2-shape ≈ 0.53 (+0.018, 11 of 12 pairs). The same three games were used to develop v2. | Good evidence, but on games that are no longer fresh, with synthetic noise. Its ideas (tight control radius, mid-gap switch placement, formation cue) are worth porting. |
| Tejesh's Viterbi | **None.** Defaults were set from physics priors and silver labels on the provided match. | Sound design and the most independent signal, but unproven. It cannot be chosen over controlled on evidence. |
| v1 / incumbent | Superseded by responsive and then controlled on IDSSE. | Retire. |
| Mike's tuned windows | Public window only. | Do not use. |

Caveats on controlled's numbers, from Valentin's own report: IDSSE multicamera tracking is cleaner
than the competition ATD (the ball is rarely missing), two games share a team, and Smart Tagging
does not exist on IDSSE — so **the tag weight (1.25) and the ball-gap behaviour are not validated by
those numbers.** Leo's degradation module is the missing piece there.

## 4. What the team learned (worth keeping regardless of model)

* **Team evidence is the bottleneck, not the decoder** (Leo's oracle budget): perfect team evidence
  on visible-ball frames lifts spellF1 from 0.43 to 0.76; perfect dead-ball evidence only to 0.50.
* **Ownership is a coin flip beyond about 4 m** of the ball; within 1 m it is right 86–99% of the
  time. Evidence should be concentrated close to the ball (v2, controlled and Tejesh's 1.5 m kernel
  all do this now).
* **Everyone under-segments.** Every model predicts fewer, longer possessions than the truth
  (Leo: about 0.65 of the true count; short possessions under 3.5 s, about 40% of all, are almost
  never matched). This is the largest remaining gap.
* **Team shape fills ball gaps.** Three independent findings agree: Tejesh's shape regression (84%
  cross-half accuracy on silver labels), Valentin's v2-shape (+0.018 on Metrica) and the
  GTT-TalentArena fork's formation model. Width is the strongest cue.
* **Broadcast sync does not help yet.** The camera check on `research/fork-scout-shape` found 0 of
  65,971 tactical ball-gap frames recoverable from the broadcast feed, with only 1,681 frames
  cross-checkable (median error 0.96 m). Drop it unless someone rebuilds the alignment.
* **Rejected with evidence:** adaptive transition costs (experiment 004, below gate, slower);
  stronger fast-ball gating and geometry smoothing (Leo, neutral or worse).
* **Promising but below gate:** the ball-contact cue (experiment 005, +0.015 on both development
  games).

## 5. How to refine this into one final solution

**Base: Valentin's `controlled` model.** Bring in the others' ideas one experiment at a time, each
judged on other-match data under a protocol written down *before* running it.

1. **Make an integration branch** (for example `final`) from `dev_VG`; keep personal branches as
   they are. Nothing needs to go to `main` until the team decides.
2. **Fix the evaluation set first.**
   * Development: IDSSE J03WMX and J03WOH, **with Leo's `validation/degrade.py` applied** so ball
     gaps, noise and dropouts look like the competition ATD. Without it, gap-handling changes cannot
     be measured. Keep the clean-tracking numbers too.
   * Smart Tagging: use Leo's synthetic tag models (`validation/tags.py`) to test the tag weight;
     today it is unvalidated.
   * Final: IDSSE J03WN1, J03WOY, J03WPY and J03WQQ are consumed. Check how many IDSSE games remain
     unseen (the dataset has seven, by its description) and keep at least one for a single final
     check. The Metrica games were used to develop v2 and are no longer clean either.
   * Never select on the public 5 minutes.
3. **Candidate experiments, in order of expected value:**
   1. **Short possessions / under-segmentation.** Lower the H/A cost or the minimum state duration
      jointly with the contact cue (005) so that real short spells survive; gate on spellF1 *and*
      unmatched predictions.
   2. **Formation cue in ball gaps** (v2-shape, or Tejesh's 7-feature version), only on frames with
      no ball, measured on degraded IDSSE.
   3. **Dead-ball pieces from Tejesh's model** that controlled may lack: "last seen out stays out"
      through gaps, forced D before each half's detected kick-off, set-piece-start dead prior.
      Compare against controlled's movement mixture rather than stacking both blindly.
   4. **Evidence fusion with Tejesh's model.** Their 16–18% disagreement means averaging the two
      models' team evidence before decoding may help; test it last, as one experiment.
4. **Consolidate the repository on `final`:** one package and one CLI (`controlled` as the
   default), one test suite (merge the 115 and the 47 tests), Tejesh's docs and website updated with
   the final model, and one technical document that lists **every external dataset** (IDSSE with its
   CC BY 4.0 attribution, the Metrica sample games) as validation data and credits the public-fork
   ideas that were reimplemented.
5. **Regenerate and freeze:** pinned requirements, `Mes_Que_Un_Code.csv` from the final model,
   validator pass, SHA-256 in the technical document, and the submission form pointing at the
   branch that holds exactly this code.

## 6. What this review did not do

* It did not re-run Valentin's IDSSE or Leo's Metrica evaluations: they need external downloads
  (figshare and GitHub), which were not fetched. The other-match numbers above are quoted from
  their reports; the models themselves were re-run and reproduced byte for byte.
* It did not crawl the 189 public forks again; section 4 relies on Valentin's scout.
* No quality ranking on the provided match is possible without the private ground truth.
