# Our solution

This directory contains our work separately from the provided challenge code and reference
outputs.

## Directory layout

```text
solution/
  code/       our implementation and supporting scripts
  results/    our frame,label submission CSV files
  reports/    generated per-submission evaluation reports
  scoreboard.csv
```

The reference implementation remains in [`../starter/`](../starter/) and the reference evaluation
artifacts remain in [`../evaluation/`](../evaluation/). This keeps the two sets of results easy to
compare and prevents our generated files from replacing the published baselines.

## Workflow

1. Put implementation code in `solution/code/`.
2. Write each submission as a `.csv` file in `solution/results/`.
3. Validate a submission:

   ```bash
   python evaluation/validate_submission.py solution/results/our_submission.csv
   ```

4. Score all our submissions and write the scoreboard and reports here:

   ```bash
   python evaluation/run_evaluation.py --results solution/results --out solution
   ```

5. Run the reference evaluation separately when needed:

   ```bash
   python evaluation/run_evaluation.py
   ```

The solution CSV files, scoreboard, and generated reports are intentionally ignored by Git. Keep
any result that should be shared in a separate tracked artifact or commit it explicitly after
checking that it contains no private ground truth.
