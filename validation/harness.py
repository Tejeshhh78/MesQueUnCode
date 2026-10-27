"""Score possession variants on degraded OTHER games with the official metric code."""

import numpy as np

from evaluation import possession_metrics as pm
from possession.data import Features
from possession.model import Config, predict
from starter.baseline import smooth

from . import degrade as degradation
from . import labels as labelling
from . import metrica

GAP = 250  # synthetic X frames between periods, so no spell or window can straddle halftime


def _speeds(xy, pitch, fps):
    metres = xy * np.asarray(pitch)
    speed = np.full(xy.shape[:2], np.nan)
    speed[2:-2] = np.linalg.norm(metres[4:] - metres[:-4], axis=2) * fps / 4
    return speed


def build_features(ball, home, away, pitch, fps):
    """The same per-frame summaries possession.data extracts from a real ATD file."""
    n = len(ball)
    players = np.concatenate([home, away], axis=1)
    team = np.r_[np.zeros(home.shape[1], dtype=int), np.ones(away.shape[1], dtype=int)]
    visible = np.isfinite(players).all(axis=2)
    speeds = _speeds(players, pitch, fps)
    valid_speed = visible & np.isfinite(speeds) & (speeds >= 0) & (speeds <= 15)
    counts = valid_speed.sum(axis=1)
    with np.errstate(invalid="ignore"):
        mean_speed = np.where(counts > 0, np.where(valid_speed, speeds, 0).sum(axis=1)
                              / np.maximum(counts, 1), np.nan)
        fast = np.where(counts > 0, (valid_speed & (speeds > 2)).sum(axis=1)
                        / np.maximum(counts, 1), np.nan)
    delta = (players - ball[:, None, :]) * np.asarray(pitch)
    distance_all = np.where(visible, np.hypot(delta[..., 0], delta[..., 1]), np.inf)
    ball_visible = np.isfinite(ball).all(axis=1)
    distance = np.full((n, 2), np.nan)
    player_speed = np.full((n, 2), np.nan)
    for side in (0, 1):
        masked = np.where(team[None, :] == side, distance_all, np.inf)
        nearest = np.argmin(masked, axis=1)
        minimum = masked[np.arange(n), nearest]
        ok = ball_visible & np.isfinite(minimum)
        distance[ok, side] = minimum[ok]
        player_speed[ok, side] = speeds[np.flatnonzero(ok), nearest[ok]]
    ball3 = np.column_stack([ball, np.zeros(n)])
    return Features(ball3, distance, player_speed, mean_speed, fast,
                    visible.sum(axis=1).astype(np.int16), ("Home", "Away"))


def timeline(game, codes, ball, home, away):
    """Concatenate the periods with an X gap; return (features, gt_codes, blocks)."""
    pieces, gt, blocks, cursor = [], [], [], 0
    for period, first, last in game.periods:
        if cursor:
            pieces.append((np.full((GAP, 2), np.nan), np.full((GAP,) + home.shape[1:], np.nan),
                           np.full((GAP,) + away.shape[1:], np.nan)))
            gt.append(np.full(GAP, labelling.X, dtype=np.int8))
            cursor += GAP
        span = slice(first, last + 1)
        pieces.append((ball[span], home[span], away[span]))
        gt.append(codes[span])
        blocks.append((period, cursor, cursor + last - first))
        cursor += last - first + 1
    ball_all, home_all, away_all = (np.concatenate(parts) for parts in zip(*pieces))
    features = build_features(ball_all, home_all, away_all, game.pitch, game.fps)
    return features, np.concatenate(gt), blocks


def baseline_labels(features, window=50):
    """starter/baseline.py on the same features: nearest team, D without a ball, majority vote."""
    d = features.distance
    codes = np.full(len(d), -1, dtype=np.int8)
    seen = np.isfinite(d).any(axis=1)
    codes[seen] = np.where(np.nan_to_num(d[seen, 0], nan=np.inf) <= np.nan_to_num(d[seen, 1], nan=np.inf), 0, 1)
    return smooth(codes, window)


def model_labels(features, blocks, pitch, fps, config=None, events=()):
    labels, _ = predict(features, list(events), blocks, pitch, fps, config or Config())
    return labels


def score(gt_codes, labels):
    pred = pm.label_codes(labels)
    result = pm.evaluate(gt_codes, pred)
    d = result["per_class"]["D"]
    t12, t25 = result["transitions"][12], result["transitions"][25]
    return {
        "spellF1": result["spell_f1_mean"], "macroF1": result["macro_f1"],
        "f1@0.5": result["spell_f1_at_headline"],
        "n_pred": result["n_pred_spells"], "n_gt": result["n_gt_spells"],
        "D_f1": d["f1"], "D_prec": d["precision"], "D_rec": d["recall"],
        "team_acc": result["team_acc_both_alive"],
        "trans12_f1": t12["f1"], "trans25_f1": t25["f1"],
        "lat25": t25["median_latency"],
        **{f"f1@{t}": v["f1"] for t, v in result["spell_f1"].items()},
    }


def prepared(game_ids=(1, 2, 3), seeds=(0,), settings=degradation.Degradation(), data_dir=None):
    """Yield (game_id, seed, features, gt_codes, blocks, game) for every combination."""
    for game_id in game_ids:
        game = metrica.load(game_id, data_dir)
        codes, _ = labelling.derive(game)
        for seed in seeds:
            ball, home, away = degradation.degrade(game, codes, settings, seed=1000 * game_id + seed)
            features, gt, blocks = timeline(game, codes, ball, home, away)
            yield game_id, seed, features, gt, blocks, game
