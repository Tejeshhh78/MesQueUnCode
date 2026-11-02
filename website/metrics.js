/*
 * Numbers shown in the Metrics section. Every value here was measured locally or is published by
 * the organisers -- nothing is estimated. Regenerate with:
 *
 *   python solution/experiments.py
 *   python evaluation/run_evaluation.py --bootstrap 1000
 *
 * `public` = the 5-minute public window (17 possessions). A pipeline check, NOT a ranking:
 *            EVALUATION.md shows it orders even the baselines wrongly. Intervals are the scorer's
 *            5-95% bootstrap over 1000 draws.
 * `fullMatch` = all 158,857 graded frames. The ranking metric needs the private ground truth,
 *            so spellF1 there is known only for the baseline (published by the organisers).
 *            Possession counts need no labels and were measured locally.
 */
window.MQUC_METRICS = {
  measuredOn: "2026-11-02",
  systems: [
    {
      id: "baseline_raw", label: "Baseline, raw", kind: "baseline",
      public: { macroF1: 0.5972, spellF1: 0.0378, spellCI: [0.019, 0.059], poss: 131 },
      fullMatch: { spellF1: 0.0486, possessions: 1949, meanPossS: 1.9, shareD: 40.7 }
    },
    {
      id: "baseline_smooth50", label: "Baseline, ±50 vote (published floor)", kind: "baseline",
      public: { macroF1: 0.6207, spellF1: 0.3256, spellCI: [0.214, 0.447], poss: 26 },
      fullMatch: { spellF1: 0.2433, possessions: 594, meanPossS: 7.2, shareD: 32.3 }
    },
    {
      id: "baseline_smooth125", label: "Baseline, ±125 vote", kind: "baseline",
      public: { macroF1: 0.5812, spellF1: 0.4242, spellCI: [0.267, 0.594], poss: 16 },
      fullMatch: { spellF1: 0.2242, possessions: 398, meanPossS: 12.4, shareD: 22.3 }
    },
    {
      id: "mes_que_un_code", label: "Mes Que Un Code (ours)", kind: "ours",
      public: { macroF1: 0.8344, spellF1: 0.4714, spellCI: [0.257, 0.671], poss: 11 },
      fullMatch: { spellF1: null, possessions: 202, meanPossS: 18.9, shareD: 39.9 }
    }
  ],
  publicTruth: { possessions: 17, meanPossS: 14.2 }
};
