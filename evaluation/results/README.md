# Put your submissions here

Every `*.csv` in this folder is scored by `run_evaluation.py`, and the file name becomes the name
on the board — so call them something you will recognise.

```bash
python evaluation/run_evaluation.py         # scores whatever is in here
python starter/baseline.py                  # regenerates the four baselines
python starter/make_dummy_submission.py     # three trivial ones, to see the loop work
```

## Generated comparison outputs

Four baselines are generated locally by `starter/baseline.py` — nearest tracked player to the ball,
then a majority vote over a window of `±0`, `±25`, `±50` and `±125` frames. `baseline_smooth50` is
the published floor.

They are worth scoring before you write anything, because the four differ in exactly one number
and the board still reorders them. That is the clearest demonstration in the repository of what
`spellF1` measures and `macroF1` does not — see [EVALUATION.md](../../docs/EVALUATION.md).

## Why these are not committed

`*.csv` here is **gitignored**. A submission is a complete set of labels for the match, and a
correct one is a copy of the ground truth. Comparison outputs are reproducible generated artifacts,
so they stay local and do not need to be committed.
