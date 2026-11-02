# MesQueUnCode possession model

Run from the repository root with the licensed tactical ATD and Smart Tagging files under `data/` or set `MTAH_DATA_DIR`.

```powershell
.\.venv\Scripts\python.exe -m pip install -r solution/requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s solution/tests
.\.venv\Scripts\python.exe -m solution.code.possession
.\.venv\Scripts\python.exe evaluation/validate_submission.py solution/results/controlled/MesQueUnCode.csv
```

The sole team model is **controlled**. It combines persistent proximity, relative ball speed, possession retention during passes and tactical evidence with a temporal decoder. The fixed team-switch cost is 12. The default run writes a CSV, manifest and technical report in `solution/results/controlled/`. Set `--team` for the registered team name and `--output-dir` for a different destination.

For aggregate diagnostics add `--diagnostics`. Run `python -m solution.code.compare_algorithms` to compare against the official starter. Review the [technical report](TECHNICAL_REPORT.md) and [independent validation procedure](INDEPENDENT_VALIDATION.md) for the model selection evidence and match split.
