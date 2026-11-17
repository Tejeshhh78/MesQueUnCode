/*
 * Numbers shown in the Metrics section. Every value was measured on EXTERNAL games or is published
 * by the organisers -- nothing is estimated, and nothing comes from the provided match.
 *
 *   python -m solution.code.calibrate_activity      (controlled, submitted)
 *   python solution/tune_external.py                (Tejesh's model, nested leave-one-game-out)
 *
 * `external` = Metrica Sports' three public sample games, labels derived from their own events,
 *              tracking degraded towards the automatic ATD, official metric code. Each model is
 *              scored on games it was not fitted or tuned on. possRatio = predicted / true
 *              possessions. See docs/EXPERIMENTS.md for the per-game table.
 * `publishedFullMatch` = the organisers' published full-match spellF1 (EVALUATION.md). Ours is
 *              scored privately by the organisers, so it stays null.
 */
window.MQUC_METRICS = {
  measuredOn: "2026-11-11",
  systems: [
    {
      id: "baseline_smooth50", label: "Organisers' baseline (±50 vote)", short: "Baseline", kind: "baseline",
      external: { spellF1: 0.2585, macroF1: 0.6558, possRatio: null },
      publishedFullMatch: 0.2433
    },
    {
      id: "v2", label: "v2 — Leo", short: "v2 (Leo)", kind: "baseline",
      external: { spellF1: 0.5095, macroF1: 0.8428, possRatio: 0.65 },
      publishedFullMatch: null
    },
    {
      id: "viterbi_shape", label: "Viterbi + team shape — Tejesh", short: "Viterbi (Tejesh)", kind: "baseline",
      external: { spellF1: 0.5396, macroF1: 0.8547, possRatio: 0.70 },
      publishedFullMatch: null
    },
    {
      id: "controlled", label: "controlled — Valentin, compliant (submitted)", short: "controlled", kind: "ours",
      external: { spellF1: 0.5432, macroF1: 0.8385, possRatio: 0.76 },
      publishedFullMatch: null
    }
  ]
};
