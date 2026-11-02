"""Generate the Mes Que Un Code possession submission from the provided match data.

    python solution/run_solution.py
    python solution/run_solution.py --window 50 --output evaluation/results/solution.csv
    python solution/run_solution.py --set switch_cost=6 --set min_spell_seconds=1.5

Reads the gated tactical bundle (ATD, Smart Tagging, homography) through games.py, so the data is
found in ./data or wherever MTAH_DATA_DIR points. Writes a frame,label CSV of exactly 199,319 rows
and validates it with the official checks before reporting success.

Deterministic: same data, same parameters, byte-identical file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from games import GAME, GAMES  # noqa: E402
from solution import config, features, loaders, possession_model, postprocess  # noqa: E402
from solution.config import Params  # noqa: E402


def predict(match: loaders.MatchData, params: Params, game: str = GAME,
            clean: bool = True) -> np.ndarray:
    """Label array (H/A/D on graded frames, X elsewhere) for one loaded match."""
    fps = GAMES[game]["fps"]
    evidence = features.build_evidence(match, params, GAMES[game]["pitch"], fps)
    states = possession_model.decode(evidence, params, match.halves)
    if clean:
        states = postprocess.clean(states, match.halves,
                                   int(round(params.min_spell_seconds * fps)),
                                   int(round(params.min_dead_seconds * fps)))
    return postprocess.to_labels(states)


def parse_overrides(pairs: Optional[List[str]]) -> Dict[str, float]:
    """['switch_cost=6', ...] -> {'switch_cost': 6.0}; ints stay ints where the default is one."""
    defaults = config.DEFAULT_PARAMS.to_dict()
    out: Dict[str, float] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError("--set expects name=value, got {!r}".format(pair))
        name, value = (part.strip() for part in pair.split("=", 1))
        if name not in defaults:
            raise KeyError("unknown parameter {!r}; see solution/config.py".format(name))
        out[name] = type(defaults[name])(float(value)) if isinstance(defaults[name], int) \
            else float(value)
    return out


def validate(path: str, n_rows: int) -> Optional[str]:
    """The official scorer's validation, applied to the written file. None when valid."""
    return pm.validate_submission(pm.read_label_csv(path), n_rows)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default=config.DEFAULT_OUTPUT,
                        help="CSV to write (default: solution/results/mes_que_un_code.csv)")
    parser.add_argument("--window", type=int, default=None,
                        help="half-width in frames of the team-evidence smoothing window "
                             "(default {})".format(config.DEFAULT_PARAMS.evidence_window))
    parser.add_argument("--set", dest="overrides", action="append", metavar="NAME=VALUE",
                        help="override any parameter in solution/config.py; repeatable")
    parser.add_argument("--data-dir", default=None,
                        help="match bundle root (default: MTAH_DATA_DIR or ./data)")
    parser.add_argument("--no-cache", action="store_true",
                        help="re-parse the raw ATD instead of using the local .npz cache")
    parser.add_argument("--no-clean", action="store_true",
                        help="skip the spell clean-up step (for ablations)")
    parser.add_argument("--print-params", action="store_true",
                        help="print the effective parameters as JSON and exit")
    args = parser.parse_args(argv)

    try:
        overrides = parse_overrides(args.overrides)
    except (KeyError, ValueError) as error:
        parser.error(str(error))
    if args.window is not None:
        overrides["evidence_window"] = args.window
    params = config.DEFAULT_PARAMS.with_overrides(**overrides)

    if args.print_params:
        print(json.dumps(params.to_dict(), indent=2, sort_keys=True))
        return 0

    started = time.time()
    try:
        match = loaders.load_match(GAME, args.data_dir, use_cache=not args.no_cache)
    except loaders.DataUnavailableError as error:
        print("ERROR  {}".format(error), file=sys.stderr)
        return 2

    tracking = match.tracking
    print("teams      H = {}, A = {}".format(tracking.home_name, tracking.away_name))
    print("halves     {}".format(", ".join("{}: {}..{}".format(h, a, b)
                                         for h, a, b in match.halves)))
    labels = predict(match, params, clean=not args.no_clean)
    path = postprocess.write_submission(args.output, labels)

    n_rows = GAMES[GAME]["n_video_frames"]
    reason = validate(path, n_rows)
    if reason:
        print("FAIL   {} did not pass the official checks: {}".format(path, reason),
              file=sys.stderr)
        return 1

    stats = postprocess.spell_stats(labels, match.halves, GAMES[GAME]["fps"])
    print("graded     {graded_frames} frames  H {share_H:.1%}  A {share_A:.1%}  D {share_D:.1%}"
          .format(**stats))
    print("spells     {possessions} possessions, mean {mean_possession_s:.1f} s; "
          "{stoppages} dead-ball stretches".format(**stats))
    print("wrote      {}  ({} rows, valid)".format(os.path.relpath(path, ROOT), n_rows))
    print("took       {:.1f} s".format(time.time() - started))
    return 0


if __name__ == "__main__":
    sys.exit(main())
