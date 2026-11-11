# Independent evaluation

Use this workflow to evaluate other matches. The competition public scorer is
for a fixed pipeline check only. Dataset provenance and recorded quality results
are in [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).

## Dataset roles

| IDSSE match IDs | Permitted role for further work |
|---|---|
| J03WMX, J03WOH | Existing development games |
| J03WN1, J03WOY | Consumed final tests (experiment 002): historical results only; no further design or selection |
| J03WPY, J03WQQ | Consumed final tests (experiment 003): historical results only; no further design or selection |
| J03WR9 | Remaining unused public IDSSE match; verify prior use before assigning a role |

Assign complete physical matches to training, development or final test before
examining candidate scores. Repeated crops/views belong to the same group.
Reserve new final games before development; never relabel consumed tests as unseen.
The historical control-cost driver and its ledgers belong to completed
experiment 002. Preserve them. New research needs its own protocol and outputs.

## Download and import

Get original files from [IDSSE version 1](https://doi.org/10.6084/m9.figshare.28196177.v1),
retaining their filenames in `solution/reports/external/idsse/`. Data attribution:
**Deutsche Fussball Liga (DFL)**; [Bassek et al. (2025)](https://doi.org/10.1038/s41597-025-04505-y),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

| Match | Match information download | Observed positions download |
|---|---|---|
| J03WMX | [51643475](https://ndownloader.figshare.com/files/51643475) | [51643514](https://ndownloader.figshare.com/files/51643514) |
| J03WN1 | [51643472](https://ndownloader.figshare.com/files/51643472) | [51643517](https://ndownloader.figshare.com/files/51643517) |
| J03WOH | [51643478](https://ndownloader.figshare.com/files/51643478) | [51643520](https://ndownloader.figshare.com/files/51643520) |
| J03WOY | [51643481](https://ndownloader.figshare.com/files/51643481) | [51643523](https://ndownloader.figshare.com/files/51643523) |
| J03WPY | [51643487](https://ndownloader.figshare.com/files/51643487) | [51643526](https://ndownloader.figshare.com/files/51643526) |
| J03WQQ | [51643484](https://ndownloader.figshare.com/files/51643484) | [51643529](https://ndownloader.figshare.com/files/51643529) |

Downloaded files were checked against published MD5 values. The importer records
original URLs and SHA-256 hashes.

Fresh conversion, without inference or scoring:

```powershell
.\.venv\Scripts\python.exe -m solution.code.import_idsse --source-dir solution/reports/external/idsse --outdir solution/reports/external/idsse-four-match-reproduction --development J03WMX J03WOH --final-test J03WN1 J03WOY
```

This reproduces the historical split; it does not make its final games unused.
Use a fresh directory: existing manifests and converted files are not overwritten.

The adapter accepts native 25 Hz only. It converts centred metre coordinates to
normalised pitch coordinates and player speed from km/h to m/s, excludes referee
tracks and preserves observations during dead ball. Periods come from metadata.
BallStatus 0 -> D; status 1 with BallPossession 1/2 -> H/A; all other combinations
remain X. These reference fields never determine features, missingness or tags.

## Evaluate controlled

Run from the repository root after importing the external data:

```powershell
python -m solution.code.independent_evaluation solution/reports/external/idsse-four-match-v1/manifest.json --development --diagnostics --output-dir solution/reports/controlled-development-check
```

Use a new output directory. Preserve the existing split and do not reuse consumed final matches for selection. The historical experiment protocols and decisions are in [experiments](experiments/). The former research drivers can be inspected at commit `4e872015`.
