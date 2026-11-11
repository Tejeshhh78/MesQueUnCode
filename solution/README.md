# Mes Que Un Code — solution

Our possession pipeline for More than a Hack 2027, Phase 1. The organisers' `starter/`,
`evaluation/` and `games.py` are imported, never modified.

```text
solution/
  config.py            every tunable number, with the reason for its value
  loaders.py           ATD / Smart Tagging / homography -> arrays on the tactical timeline
  features.py          per-frame evidence: team control, dead ball, team shape
  possession_model.py  emission scores + three-state Viterbi decoder
  postprocess.py       spell clean-up, submission writer, spell statistics
  run_solution.py      entry point: data in, validated CSV out
  experiments.py       baselines vs solution vs ablations
  tests/               unittest suite
  results/             our submissions (*.csv gitignored)
```

## Run

```bash
python solution/run_solution.py                          # -> solution/results/mes_que_un_code.csv
python solution/run_solution.py --output evaluation/results/mes_que_un_code.csv
python solution/run_solution.py --window 12 --set switch_cost=5
python solution/run_solution.py --print-params           # effective parameters as JSON
python solution/run_solution.py --help
```

The data is found in `./data` or wherever `MTAH_DATA_DIR` points (`--data-dir` overrides both).
The first run parses the raw ATD and caches it under `.cache/mesqueuncode/` (gitignored).
The output is validated with the official checks before the script reports success. Exit codes:
0 written and valid, 1 failed validation, 2 data missing.

## Validate and score

```bash
python evaluation/validate_submission.py solution/results/mes_que_un_code.csv
python evaluation/run_evaluation.py --results solution/results --out solution
```

The second command writes `solution/scoreboard.csv` and `solution/reports/`, leaving the reference
scoreboard in `evaluation/` alone. Remember that the public window is a pipeline check, not a
tuning set — see [docs/EXPERIMENTS.md](../docs/EXPERIMENTS.md).

## External data: fitting, tuning, evaluation

Everything fitted or tuned uses external data only (the organisers' rule clarification):

```bash
python solution/calibrate.py --download --write  # fit motion + team-shape models -> calibration.json
python solution/calibrate.py --check             # confirm calibration.json is reproducible
python solution/evaluate_external.py             # whole pipeline vs baseline on external games
python solution/tune_external.py                 # decoder grid search, nested leave-one-out
```

The games (Metrica Sports' public samples) are downloaded into `data/external/`, which is
gitignored. `solution/external/` holds Leo Hofmann's loader, event-derived labels and tracking
degradation.

## Experiments and tests

```bash
python solution/experiments.py                  # provided-match comparison; keep its output local
python solution/experiments.py --write-csv      # also writes every variant to solution/results/
python -m unittest discover -s solution/tests -t .
```

Tests needing the gated data skip themselves when it is absent.

## Final submission

```bash
pip install -r requirements-solution.txt        # the pinned environment the file was built with
python solution/run_solution.py --output solution/results/Mes_Que_Un_Code.csv
python evaluation/validate_submission.py solution/results/Mes_Que_Un_Code.csv
```

Upload exactly the file you just validated, together with the technical document
[`docs/Mes_Que_Un_Code.md`](../docs/Mes_Que_Un_Code.md) (SUBMISSION.md). Rerunning the command
regenerates the CSV byte for byte; its SHA-256 is listed at the top of the technical document.

More: [ARCHITECTURE](../docs/ARCHITECTURE.md) · [MODEL](../docs/MODEL.md) ·
[EXPERIMENTS](../docs/EXPERIMENTS.md) · [TROUBLESHOOTING](../docs/TROUBLESHOOTING.md)
