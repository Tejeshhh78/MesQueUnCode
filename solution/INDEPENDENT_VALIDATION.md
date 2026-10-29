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
The existing `control_experiment.py` and its ledgers belong to completed
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

## Evaluate development data

For the already converted four-match dataset, compare incumbent with the fixed
responsive profile on its development games only:

```powershell
.\.venv\Scripts\python.exe -m solution.code.independent_evaluation solution/reports/external/idsse-four-match-v1/manifest.json --development --profiles incumbent responsive --bootstrap 1000 --output-dir solution/reports/independent-development-check
```

New experiments should instead compare their candidate against frozen responsive
12. Extend the existing profile/configuration plumbing without changing the
official scorer. The generic evaluator defaults to incumbent/reliable unless
profiles are specified. Development evaluation does not open unselected final
feature or label files. Always use a new output directory.

Inputs may be ATD, numeric NPZ or JSON features; inline arrays are synthetic only.
The [synthetic manifest](tests/fixtures/independent_manifest.synthetic.json)
documents schema version 1: canonical match/group identities, split, evidence
kind, pitch, 25 fps, units, frame coverage, half-open period blocks, input paths,
reference paths and provenance. Provenance must describe source, licence,
organiser access, label method/author and whether inputs use reference labels.
Declarations and path/hash guards catch mistakes; they do not prove independence.

The scorer receives an evaluation grid with 51 ignored X frames inserted only
between metadata periods. All source frames and within-period unknown gaps are
preserved; inference uses the original grid. This prevents possessions merging
across halftime or transitions matching across periods without changing official
metric code. Unknown provider-reference frames are never filled as invented truth.

Results separate reference/proxy/synthetic evidence and contain per-match official
metrics, paired differences, configuration, environment and source/input/prediction
hashes. Prediction timing excludes loading and copying; do not equate it with
full CLI timing. Within-match bootstrap intervals do not measure between-match
uncertainty or establish a paired gain.

## Final-test discipline

Select on development first; freeze the candidate, configuration, code, inputs
and acceptance rules. Score it once on genuinely new final matches. Do not
change a candidate after examining those results and continue calling the same
matches held-out. Keep failed and neutral experiments as well as successful ones.

Completed experiment 002 is recorded locally under
`solution/reports/quality-development/control-switch-cost-002/` and
`control-switch-cost-002-final/`, with the evaluated-source snapshot. Its tracked
[protocol](experiments/control-switch-cost-002.json) is immutable history, not
an instruction to repeat selection or reset consumption markers.

Raw inputs, converted labels, predictions and detailed traces stay in ignored
local directories. Keep source/configuration hashes and existing artifacts when
cleaning documentation; do not discard the evidence behind published claims.
