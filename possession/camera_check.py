"""Check whether the broadcast tracking can recover tactical ball gaps.

Run python -m possession.camera_check from the repository root. Reports only aggregates.
"""
import argparse
import json

import numpy as np

from games import GAME, GAMES, half_video_blocks, require, source_frames


def load(source, data_dir):
    path, = require(["atd_raw"], data_dir=data_dir, source=source)
    n = source_frames(source)
    pitch = np.array(GAMES[GAME]["pitch"])
    shape = np.full((n, 4), np.nan)
    ball = np.full((n, 2), np.nan)
    count = 0
    with open(path, encoding="utf-8-sig") as stream:
        for i, line in enumerate(stream):
            frame, players, point = line.strip().split(":")
            if i >= n or int(frame) != i + 1:
                raise ValueError(f"{source}: unexpected frame {frame}")
            xy = np.fromstring(players.rstrip(";").replace(";", ","), sep=",").reshape(-1, 3)[:, :2]
            xy = xy[np.isfinite(xy).all(axis=1) & ((xy >= 0) & (xy <= 1)).all(axis=1)]
            if len(xy) >= 10:
                xy = (xy - 0.5) * pitch
                shape[i] = [*xy.mean(axis=0), *xy.std(axis=0)]
            coords = np.fromstring(point.rstrip(";"), sep=",")[:2]
            if len(coords) == 2 and np.isfinite(coords).all() and ((coords >= -0.05) & (coords <= 1.05)).all():
                ball[i] = (coords - 0.5) * pitch
            count += 1
    if count != n:
        raise ValueError(f"{source}: expected {n} frames, found {count}")
    return shape, ball


def align(tactical, broadcast, fps):
    """Find unique five-second blocks in player geometry, then refine to frame resolution."""
    width = 5 * fps
    offsets = np.linspace(0, width - 1, 6, dtype=int)
    starts = np.arange(0, len(tactical) - width + 1, fps)
    windows = tactical[starts[:, None] + offsets]
    matches = {}
    for b in range(0, len(broadcast) - width + 1, width):
        query = broadcast[b + offsets]
        if not np.isfinite(query).all():
            continue
        candidates = []
        for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            sign = np.array([sx, sy, 1, 1])
            error = np.mean((windows - query * sign) ** 2, axis=(1, 2))
            error = np.where(np.isfinite(error), error, np.inf)
            best = int(np.argmin(error))
            candidates.append((error[best], int(starts[best]), sx, sy))
            far = np.abs(starts - starts[best]) > width
            if far.any():
                candidates.append((error[far].min(), -1, sx, sy))
        candidates.sort()
        cost, coarse, sx, sy = candidates[0]
        if coarse < 0 or cost >= 2.5 ** 2 or cost >= 0.5 * candidates[1][0]:
            continue
        fine = np.arange(max(0, coarse - fps), min(len(tactical) - width, coarse + fps) + 1)
        error = np.mean((tactical[fine[:, None] + offsets] - query * [sx, sy, 1, 1]) ** 2, axis=(1, 2))
        best = int(np.argmin(np.where(np.isfinite(error), error, np.inf)))
        matches[b] = (int(fine[best]), sx, sy)
    # Isolated matches near cuts are especially prone to accidental alignment.
    for b, (t, sx, sy) in matches.items():
        for step in (-width, width):
            other = matches.get(b + step)
            if other and other[1:] == (sx, sy) and abs(other[0] - t - step) <= fps // 2:
                yield b, t, sx, sy
                break


def evaluate(tactical, broadcast, fps, blocks):
    shape, truth = tactical
    shape_b, ball_b = broadcast
    active = np.zeros(len(shape), dtype=bool)
    for _, start, last in blocks:
        active[start:last + 1] = True
    shape = np.where(active[:, None], shape, np.nan)
    proposed = np.full_like(truth, np.nan)
    assignments = np.zeros(len(truth), dtype=int)
    width = fps * 5
    for b, t, sx, sy in align(shape, shape_b, fps):
        proposed[t:t + width] = ball_b[b:b + width] * [sx, sy]
        assignments[t:t + width] += 1
    usable = active & (assignments == 1) & np.isfinite(proposed).all(axis=1)
    observed = np.isfinite(truth).all(axis=1)
    errors = np.linalg.norm(truth[usable & observed] - proposed[usable & observed], axis=1)
    return {"active_frames": int(active.sum()), "missing_tactical_frames": int((active & ~observed).sum()),
            "candidate_gap_frames": int((usable & ~observed).sum()), "overlap_check_frames": len(errors),
            "median_error_m": float(np.median(errors)) if len(errors) else None,
            "p90_error_m": float(np.quantile(errors, 0.9)) if len(errors) else None,
            "within_3m_fraction": float(np.mean(errors <= 3)) if len(errors) else None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir")
    args = parser.parse_args()
    print(json.dumps(evaluate(load("tactical", args.data_dir), load("broadcast", args.data_dir),
                              GAMES[GAME]["fps"], half_video_blocks(GAME)), indent=2))
