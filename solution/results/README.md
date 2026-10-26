# Our submission results

Place our `frame,label` submission CSV files in this directory. Every CSV here is scored by:

```bash
python evaluation/run_evaluation.py --results solution/results --out solution
```

Validate each file before scoring:

```bash
python evaluation/validate_submission.py solution/results/our_submission.csv
```

Generated submission CSV files are ignored by Git. The committed reference submissions are in
[`../../evaluation/results/`](../../evaluation/results/).
