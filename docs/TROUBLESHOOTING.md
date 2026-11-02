# Troubleshooting

## `ERROR  match data not found`

`solution/run_solution.py` and `solution/experiments.py` need the gated tactical bundle. The message
lists every missing file and where it was expected.

* Unpack the bundle into `data/` keeping its folder names (`data/tactical/…`), **or**
* point `MTAH_DATA_DIR` at it:

  ```bash
  export MTAH_DATA_DIR=/path/to/mtah2027        # macOS / Linux
  $env:MTAH_DATA_DIR = "D:\mtah2027"            # Windows PowerShell
  ```

  or pass `--data-dir /path/to/mtah2027` for one run.

Exit code 2 means "data missing". Without the data you can still run the scorer, the validator and
the synthetic tests:

```bash
python evaluation/run_evaluation.py
python -m unittest discover -s solution/tests -t .
```

## `ModuleNotFoundError: No module named 'solution'`

Run the scripts from the repository root (`python solution/run_solution.py`), or as modules
(`python -m solution.run_solution`). The scripts add the root to `sys.path` themselves; running a
module file from inside `solution/` with `-m` from another directory will not.

## The first run is slow; later runs are fast

The first run parses the ~200 MB ATD text file (≈10 s) and writes a compressed cache to
`.cache/mesqueuncode/`. It is keyed on the source file's size and modification time, so a new
bundle is re-parsed automatically. Force a re-parse with `--no-cache`, or move the cache with
`MQUC_CACHE_DIR`. The cache is derived match data: it is gitignored and must never be committed.

## `FAIL … row count 164488 != 199319`

You indexed the broadcast recording. The submission is indexed to the tactical video; see
[ARCHITECTURE.md](ARCHITECTURE.md#tactical-versus-broadcast).

## `FAIL … 'frame' column is not 0..199318 in order`

An off-by-one between ATD `frame_id` (starts at 1) and video frames (start at 0). The solution's
writer always numbers rows `0…199318`; if you post-process a CSV yourself, keep that.

## `rebuilt baseline_… differs from evaluation/results/…`

`solution/experiments.py` compares its vectorised nearest-player step with the CSVs
`starter/baseline.py` wrote. A mismatch means one of them was produced from different data or
code — regenerate the baselines with `python starter/baseline.py` and rerun.

## The scorer's board is missing my file

`run_evaluation.py` scores `evaluation/results/*.csv` by default. Our default output is
`solution/results/mes_que_un_code.csv`; score it with

```bash
python evaluation/run_evaluation.py --results solution/results --out solution
```

or write to `evaluation/results/` with `--output`.

## Results differ between machines

They should not: the pipeline is deterministic. Check the data bundle (file sizes), the parameter
set (`python solution/run_solution.py --print-params`), and that no `--set` overrides were passed.
numpy/pandas versions that change float parsing could in principle move a decision on a tie; the
measured results in [EXPERIMENTS.md](EXPERIMENTS.md) state the versions used.

## Website shows no team or metrics

`team.js` and `metrics.js` must load before `app.js`. Check the browser console for a syntax error
in an edited `team.js`, e.g. a missing comma between entries.
