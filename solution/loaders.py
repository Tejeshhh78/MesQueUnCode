"""Read the provided match assets into arrays indexed by tactical video frame.

Three inputs, all on the tactical timeline:

  * ATD (tracking)    frame_id 1..199319  ->  video_frame = frame_id - 1
  * Smart Tagging     seconds from video start  ->  video_frame = round(seconds * 25)
  * homography        frame 0..199318, already a video frame

The per-half offsets in games.py map video frames to the hidden IdFrame counter the ground truth
was built on. The ATD we are given is already on the video timeline, so they are never added here;
what we take from games.py is the graded half boundaries (half_video_blocks), which are published.

The raw ATD is ~200 MB of text. The starter baseline streams it line by line in pure Python; here it
is parsed once with pandas' C reader and cached as a compressed .npz keyed on the source file's size
and mtime, so the second run takes a couple of seconds. The cache is local and gitignored -- it is
derived match data and must never be committed.
"""

from __future__ import annotations

import io
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from solution import config

sys.path.insert(0, config.ROOT)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from starter.baseline import team_names, team_of_each_track  # noqa: E402
from starter import load_homography, load_smart_tagging  # noqa: E402

HOME, AWAY = 0, 1


class DataUnavailableError(RuntimeError):
    """The gated match bundle is not where games.py looks for it."""


@dataclass(frozen=True)
class Tracking:
    """Positions in metres on the tactical timeline. NaN wherever the tracker saw nothing.

    players  (n_frames, n_slots, 3) float32: x, y, speed (m/s)
    ball     (n_frames, 3)          float32: x, y, speed (m/s; often NaN even when x, y exist)
    team     (n_slots,)             int8:    HOME or AWAY for each track slot
    """
    players: np.ndarray
    ball: np.ndarray
    team: np.ndarray
    home_name: str
    away_name: str
    # +1 if the home team attacks towards increasing x in the first half, -1 otherwise. Teams swap
    # ends at halftime, so the second half is the opposite sign.
    home_attack_sign: int = 1

    def home_direction(self, half: int) -> int:
        return self.home_attack_sign if half == 1 else -self.home_attack_sign

    @property
    def n_frames(self) -> int:
        return int(self.ball.shape[0])


@dataclass(frozen=True)
class TagEvent:
    """One Smart Tagging instance, as a half-open video-frame interval."""
    code: str
    start_frame: int
    end_frame: int
    team: Optional[int]          # HOME, AWAY, or None for "N/A" and anything unrecognised
    set_piece_type: Optional[str]  # lower-cased "Type" tag, e.g. "throw-in", or None


@dataclass(frozen=True)
class MatchData:
    tracking: Tracking
    tags: List[TagEvent]
    homography_error: Optional[np.ndarray]  # (n_frames,) reprojection error, NaN where invalid
    halves: List[Tuple[int, int, int]]      # [(half, first_video_frame, last_video_frame)]


def check_data_available(game: str = GAME, data_dir: Optional[str] = None) -> None:
    """Raise DataUnavailableError with setup instructions if the tactical bundle is missing."""
    try:
        require(["atd_meta", "atd_raw", "smart_tagging"], game, data_dir)
    except IOError as error:
        raise DataUnavailableError(
            "{}\n\nThe solution needs the gated tactical ATD and Smart Tagging. The public scorer "
            "still works without them:\n    python evaluation/run_evaluation.py".format(error)
        ) from None


# -- ATD ------------------------------------------------------------------------------------------
def parse_raw_atd(text: str, n_slots: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(frame_ids, players, ball) from raw FIFA/EPTS text.

    A line is "<frame_id>:<x,y,s>;<x,y,s>;...:<x,y,s>". Turning both group separators into commas
    makes every line one flat CSV row of 1 + 3 * n_slots + 3 numbers, which pandas parses in C.
    """
    flat = text.replace(":", ",").replace(";", ",")
    table = pd.read_csv(io.StringIO(flat), header=None, dtype=np.float64,
                        na_values=["NaN"], keep_default_na=False)
    expected = 1 + 3 * n_slots + 3
    if table.shape[1] != expected:
        raise ValueError("ATD rows have {} fields, expected {} for {} track slots".format(
            table.shape[1], expected, n_slots))
    values = table.to_numpy()
    frame_ids = values[:, 0].astype(np.int64)
    players = values[:, 1:1 + 3 * n_slots].reshape(len(values), n_slots, 3).astype(np.float32)
    ball = values[:, 1 + 3 * n_slots:].astype(np.float32)
    return frame_ids, players, ball


def _to_video_timeline(frame_ids: np.ndarray, players: np.ndarray, ball: np.ndarray,
                       n_frames: int) -> Tuple[np.ndarray, np.ndarray]:
    """Scatter ATD records onto video frames 0..n_frames-1. Frames with no record stay NaN."""
    video = frame_ids - 1
    if video.min() < 0 or video.max() >= n_frames:
        raise ValueError("ATD frame_id range {}..{} does not fit a {}-frame video".format(
            int(frame_ids.min()), int(frame_ids.max()), n_frames))
    if len(np.unique(video)) != len(video):
        raise ValueError("ATD has duplicate frame_ids")
    out_players = np.full((n_frames,) + players.shape[1:], np.nan, dtype=np.float32)
    out_ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    out_players[video] = players
    out_ball[video] = ball
    return out_players, out_ball


def home_attack_sign(meta_path: str) -> int:
    """+1 if the first team in the metadata (home) attacks left to right in the first half."""
    root = ET.parse(meta_path).getroot()
    team = next(iter(root.iter("Team")), None)
    if team is None:
        raise ValueError("{}: no Team entries".format(meta_path))
    for parameter in team.iter("ProviderParameter"):
        if (parameter.findtext("Name") or "").strip() == "attack_direction_first_half":
            value = (parameter.findtext("Value") or "").strip().lower()
            if value == "left_to_right":
                return 1
            if value == "right_to_left":
                return -1
            raise ValueError("{}: unexpected attack direction {!r}".format(meta_path, value))
    raise ValueError("{}: home team has no attack_direction_first_half".format(meta_path))


def _cache_path(raw_path: str) -> str:
    stat = os.stat(raw_path)
    stem = os.path.splitext(os.path.basename(raw_path))[0]
    return os.path.join(config.CACHE_DIR, "{}_{}_{}.npz".format(
        stem, stat.st_size, int(stat.st_mtime)))


def load_tracking(game: str = GAME, data_dir: Optional[str] = None,
                  use_cache: bool = True) -> Tracking:
    """Tactical ATD in metres, indexed by video frame."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    team = team_of_each_track(meta_path)
    home_name, away_name = team_names(meta_path)
    n_frames = source_frames("tactical", game)

    cache = _cache_path(raw_path)
    if use_cache and os.path.isfile(cache):
        with np.load(cache) as stored:
            players, ball = stored["players"], stored["ball"]
    else:
        with open(raw_path) as handle:
            frame_ids, players, ball = parse_raw_atd(handle.read(), len(team))
        players, ball = _to_video_timeline(frame_ids, players, ball, n_frames)
        if use_cache:
            os.makedirs(config.CACHE_DIR, exist_ok=True)
            np.savez_compressed(cache, players=players, ball=ball)

    length, width = GAMES[game]["pitch"]
    scale = np.array([length, width], dtype=np.float32)
    players = players.copy()
    ball = ball.copy()
    players[..., :2] *= scale
    ball[:, :2] *= scale
    return Tracking(players=players, ball=ball, team=team,
                    home_name=home_name, away_name=away_name,
                    home_attack_sign=home_attack_sign(meta_path))


# -- Smart Tagging --------------------------------------------------------------------------------
def team_code(tag_team: Optional[str], home_name: str, away_name: str) -> Optional[int]:
    """Map a Smart Tagging Team value to HOME/AWAY, case-insensitively. None for N/A or unknown."""
    if not tag_team:
        return None
    value = tag_team.strip().lower()
    if value == home_name.strip().lower():
        return HOME
    if value == away_name.strip().lower():
        return AWAY
    return None


def load_tags(home_name: str, away_name: str, game: str = GAME,
              data_dir: Optional[str] = None) -> List[TagEvent]:
    """Smart Tagging as TagEvents on the video timeline, via the starter's stdlib reader."""
    out = []
    for event in load_smart_tagging.load(game, data_dir):
        kind = event["tags"].get("Type")
        out.append(TagEvent(
            code=(event["code"] or "").strip().upper(),
            start_frame=event["start_frame"],
            end_frame=event["end_frame"],
            team=team_code(event["tags"].get("Team"), home_name, away_name),
            set_piece_type=kind.strip().lower() if kind else None,
        ))
    return out


# -- homography -----------------------------------------------------------------------------------
def load_homography_error(game: str = GAME, data_dir: Optional[str] = None) -> Optional[np.ndarray]:
    """Per-frame reprojection error (NaN where no matrix exists), or None if the file is absent."""
    try:
        require(["homography"], game, data_dir)
    except IOError:
        return None
    _, error, valid = load_homography.load(game, data_dir)
    error = np.asarray(error, dtype=np.float64).copy()
    error[~np.asarray(valid, dtype=bool)] = np.nan
    return error


def load_match(game: str = GAME, data_dir: Optional[str] = None,
               use_cache: bool = True) -> MatchData:
    """Everything the model reads, or DataUnavailableError naming what is missing."""
    check_data_available(game, data_dir)
    tracking = load_tracking(game, data_dir, use_cache)
    tags = load_tags(tracking.home_name, tracking.away_name, game, data_dir)
    return MatchData(tracking=tracking, tags=tags,
                     homography_error=load_homography_error(game, data_dir),
                     halves=list(half_video_blocks(game)))


def graded_mask(n_frames: int, halves: Sequence[Tuple[int, int, int]]) -> np.ndarray:
    """Boolean mask of frames inside a graded half (inclusive bounds from half_video_blocks)."""
    mask = np.zeros(n_frames, dtype=bool)
    for _, first, last in halves:
        mask[first:last + 1] = True
    return mask


def tag_counts(tags: Sequence[TagEvent]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for tag in tags:
        counts[tag.code] = counts.get(tag.code, 0) + 1
    return counts
