/*
 * Numbers shown in the Metrics section. Every value here was measured on EXTERNAL games or is
 * published by the organisers -- nothing is estimated, and nothing comes from the provided match.
 * Regenerate with:
 *
 *   python solution/evaluate_external.py
 *   python solution/tune_external.py
 *
 * `external` = Metrica Sports' three public sample games, labels derived from their own events,
 *              tracking degraded towards the automatic ATD, scored with the official metric code.
 *              spellF1 for the tuned model is the nested leave-one-game-out figure: each game is
 *              scored by a model whose costs were chosen on the other two. possRatio = predicted /
 *              true possessions.
 * `publishedFullMatch` = the organisers' published full-match spellF1 (EVALUATION.md). Our own is
 *              scored privately by the organisers, so it stays null.
 */
window.MQUC_METRICS = {
  measuredOn: "2026-11-04",
  systems: [
    {
      id: "baseline_smooth50", label: "Baseline, ±50 vote (published floor)", short: "Baseline ±50", kind: "baseline",
      external: { spellF1: 0.2585, macroF1: 0.6558, possRatio: null },
      publishedFullMatch: 0.2433
    },
    {
      id: "ours_priors", label: "Mes Que Un Code, hand-picked costs", short: "Ours, untuned", kind: "baseline",
      external: { spellF1: 0.4663, macroF1: 0.8486, possRatio: 0.48 },
      publishedFullMatch: null
    },
    {
      id: "ours_tuned", label: "Mes Que Un Code, tuned on external games (submitted)", short: "Mes Que Un Code", kind: "ours",
      external: { spellF1: 0.5396, macroF1: 0.8563, possRatio: 0.66 },
      publishedFullMatch: null
    }
  ]
};
