# Architecture

How the repository is laid out, how data flows through the solution, and which parts are ours
versus the organisers'.

## Repository structure

```text
games.py                     organisers: per-match constants, asset paths, half boundaries, IdFrame offsets
evaluation/                  organisers: the official scorer -- never modified by us
  possession_metrics.py      metric definitions (spellF1, macroF1, ...)
  run_evaluation.py          the scorer that produces the official result
  validate_submission.py     the exact format checks the scorer applies
  gt/                        public 5-minute ground truth (frames 11185..18684)
  results/                   submissions scored by default (gitignored *.csv)
starter/                     organisers: loaders, reference baseline, notebooks
solution/                    ours
  config.py                  every tunable parameter, with its justification
  loaders.py                 ATD / Smart Tagging / homography -> arrays on the tactical timeline
  features.py                per-frame evidence: team control, dead ball, team shape
  possession_model.py        emission scores + three-state Viterbi decoder
  postprocess.py             spell clean-up, CSV writer, spell statistics
  run_solution.py            command-line entry point
  experiments.py             baselines vs solution vs ablations (provided match; keep output local)
  calibrate.py               fits motion + team-shape models on EXTERNAL games -> calibration.json
  calibration.json           the fitted values, committed; only applied at inference
  tune_external.py           decoder grid search on EXTERNAL games (nested leave-one-game-out)
  evaluate_external.py       whole pipeline vs baseline on EXTERNAL games
  external/                  Metrica sample-game loader, event-derived labels, ATD-like degradation
  tests/                     unittest suite (synthetic + optional real-data tests)
  results/                   our submissions (gitignored *.csv)
website/                     static team site (HTML/CSS/JS, no build step)
docs/                        challenge docs (organisers) + ARCHITECTURE/MODEL/EXPERIMENTS/WEBSITE/TROUBLESHOOTING (ours)
data/                        gated match bundle, gitignored except data/README.md
data/external/               downloaded Metrica public sample games (external), gitignored
.cache/mesqueuncode/         parsed-ATD cache (.npz), gitignored, derived data -- never commit
```

`solution/` imports the organisers' code instead of duplicating it: `games.py` for paths and half
boundaries, `starter/baseline.py` for team-of-slot parsing and the majority-vote smoothing,
`starter/load_smart_tagging.py` and `starter/load_homography.py` for those formats, and
`evaluation/possession_metrics.py` for validation and scoring.

## Data flow

```text
data/tactical/*_atd.xml ─┐
data/tactical/*_atd.txt ─┼─ loaders.load_match() ──► MatchData
*_smart_tagging.xml ─────┤      Tracking: players (n,46,3) m, ball (n,3) m, team per slot
*_homography.csv ────────┘      TagEvents on video frames, homography error, graded halves
                                        │
                         features.build_evidence() ──► Evidence
                                team[f]  log-odds H vs A
                                dead[f]  log-odds D vs live
                                forced_dead[f]  before each half's kick-off
                                        │
                  possession_model.decode() ──► state per frame (Viterbi, per half)
                                        │
                   postprocess.clean() / to_labels() ──► H/A/D on graded frames, X elsewhere
                                        │
             postprocess.write_submission() ──► 199,319-row frame,label CSV
                                        │
                possession_metrics.validate_submission()  (official checks, before success)
```

## Timelines and frame alignment

There are three counters in this project. Only one of them is a submission row.

| source | index | to a submission row |
|---|---|---|
| tactical video / submission / homography | `0 … 199318` | identity |
| ATD tracking | `frame_id 1 … 199319` | `frame_id - 1` |
| Smart Tagging | seconds from video start | `round(seconds × 25)` |
| hidden IdFrame (ground-truth derivation) | `video + offsets[half]` | not needed |

`games.py` carries per-half offsets (`1702057` for the first half, `1702058` for the second) that
map video frames to the IdFrame counter the ground truth was derived from. The two halves differ by
one frame, so no single offset is correct for both. **The provided ATD is already on the video
timeline**, so those offsets are never added to tracking data; `solution/` uses
`games.half_video_blocks()` only to know which frames are graded:

| | first | last | graded |
|---|---|---|---|
| before kick-off | 0 | 11,184 | no |
| 1st half | 11,185 | 90,651 | yes |
| halftime | 90,652 | 110,807 | no |
| 2nd half | 110,808 | 190,197 | yes (D until kick-off at 113,742) |
| after the whistle | 190,198 | 199,318 | no |

Decoding runs independently inside each half, so nothing — no spell, no smoothing window, no
transition — ever crosses halftime. Ungraded frames are written as `X`.

`solution/tests/test_alignment.py` pins all of this down, and `solution/experiments.py` rebuilds
the four published baselines from the cached arrays and asserts they are byte-identical to
`starter/baseline.py`'s output — the strongest available check that the tracking join is right.

## Tactical versus broadcast

The submission is indexed to the **tactical** recording (fixed camera, 199,319 frames, homography
valid on every frame). The broadcast recording has its own 164,488-frame timeline with cuts and
replays, is not synchronised to tactical, and has no mapping table. The solution does not use it:
mapping it back would need a video-synchronisation step whose errors would land directly on
transition timing. A ~164k-row submission means the broadcast timeline was indexed by mistake, and
the validator rejects it.

## Determinism

No randomness anywhere in the pipeline. The Viterbi breaks ties by a fixed rule (stay, then the
lowest state index), the logistic regression is solved by Newton's method from zero, and the CSV
writer always uses `\n`. Same data and parameters give a byte-identical file;
`test_writing_is_byte_deterministic` and `test_prediction_is_deterministic` check it.
