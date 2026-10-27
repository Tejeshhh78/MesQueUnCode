"""Loaders for Metrica Sports' public sample games (https://github.com/metrica-sports/sample-data).

Those three anonymised matches are OTHER games with synchronised tracking and event data, so they
may be used to validate and tune (RULES.md: labelled data from other games is fine). Download them
into data/external/ (gitignored) with ``python -m validation --download``.

Every game is returned in one in-memory form:

    Game(name, fps, pitch, home_xy, away_xy, ball_xy, periods, events)

``*_xy`` are normalised 0-1 coordinates with NaN where absent, indexed by 0-based frame.
``periods`` is [(period, first, last)] with inclusive 0-based frame bounds. ``events`` is a list of
dicts {team: 0 home / 1 away / None, type, subtypes, start, end, period} on the same frame index.
"""

from dataclasses import dataclass
import io
import json
from pathlib import Path
import re
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "data" / "external"
SOURCE = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
FILES = {
    1: ["Sample_Game_1_RawEventsData.csv", "Sample_Game_1_RawTrackingData_Home_Team.csv",
        "Sample_Game_1_RawTrackingData_Away_Team.csv"],
    2: ["Sample_Game_2_RawEventsData.csv", "Sample_Game_2_RawTrackingData_Home_Team.csv",
        "Sample_Game_2_RawTrackingData_Away_Team.csv"],
    3: ["Sample_Game_3_events.json", "Sample_Game_3_metadata.xml", "Sample_Game_3_tracking.txt"],
}
PITCH = (105.0, 68.0)  # stated by the sample-data README for all three games
FPS = 25


@dataclass
class Game:
    name: str
    fps: int
    pitch: tuple
    home_xy: np.ndarray
    away_xy: np.ndarray
    ball_xy: np.ndarray
    periods: list
    events: list


def folder(game, data_dir=None):
    return Path(data_dir or DEFAULT_DIR) / f"metrica_sample_game_{game}"


def download(games=(1, 2, 3), data_dir=None):
    """Fetch the public files once. Network access only here, and only from the public repo."""
    for game in games:
        target = folder(game, data_dir)
        target.mkdir(parents=True, exist_ok=True)
        stem = f"Sample_Game_{game}"
        for name in FILES[game]:
            path = target / name
            if not path.is_file():
                print(f"downloading {name} ...", flush=True)
                urllib.request.urlretrieve(f"{SOURCE}/{stem}/{name}", path)


def _subtypes(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ()
    if isinstance(value, dict):
        return (value["name"].upper(),)
    if isinstance(value, list):
        return tuple(item["name"].upper() for item in value)
    return tuple(part.strip().upper() for part in str(value).split("-") if part.strip())


def _csv_tracking(path):
    """Metrica CSV tracking: three header rows, then Period, Frame, Time, (x, y) per player, ball."""
    table = pd.read_csv(path, header=None, skiprows=3, low_memory=False).to_numpy(dtype=np.float64)
    frames = table[:, 1].astype(np.int64)
    if not np.array_equal(frames, np.arange(1, len(frames) + 1)):
        raise ValueError(f"{path.name}: frames are not 1..n")
    players = table[:, 3:-2].reshape(len(table), -1, 2)
    return table[:, 0].astype(np.int64), players, table[:, -2:]


def _load_csv_game(game, data_dir):
    base = folder(game, data_dir)
    events_name, home_name, away_name = FILES[game]
    period, home, ball = _csv_tracking(base / home_name)
    period_away, away, ball_away = _csv_tracking(base / away_name)
    if not np.array_equal(period, period_away):
        raise ValueError("home/away tracking periods disagree")
    same = np.isfinite(ball) & np.isfinite(ball_away)
    if not np.allclose(ball[same], ball_away[same]):
        raise ValueError("home/away ball tracks disagree")
    periods = [(int(p), int(np.flatnonzero(period == p)[0]), int(np.flatnonzero(period == p)[-1]))
               for p in np.unique(period) if p > 0]
    raw = pd.read_csv(base / events_name)
    events = []
    for row in raw.itertuples(index=False):
        team = {"home": 0, "away": 1}.get(str(row.Team).strip().lower())
        start, end = int(row[4]) - 1, int(row[6]) - 1
        events.append({"team": team, "type": str(row.Type).strip().upper(),
                       "subtypes": _subtypes(row.Subtype), "start": start,
                       "end": max(start, end), "period": int(row.Period)})
    return Game(f"metrica_{game}", FPS, PITCH, home, away, ball, periods, events)


def _strip(tag):
    return re.sub(r"\{.*\}", "", tag)


def _load_epts_game(game, data_dir):
    """Metrica FIFA/EPTS: per-frame-range player order (substitutions), x/y only, ball x/y."""
    base = folder(game, data_dir)
    events_name, meta_name, raw_name = FILES[game]
    root = ET.parse(base / meta_name).getroot()
    for element in root.iter():
        element.tag = _strip(element.tag)
    team_ids = [team.get("id") for team in root.iter("Team")]
    player_team = {player.get("id"): team_ids.index(player.get("teamId"))
                   for player in root.iter("Player")}
    channel_player = {channel.get("id"): channel.get("playerId")
                      for channel in root.iter("PlayerChannel")}
    params = {p.findtext("Name"): p.findtext("Value") for p in root.iter("ProviderParameter")}
    home_ids = sorted(p for p, t in player_team.items() if t == 0)
    away_ids = sorted(p for p, t in player_team.items() if t == 1)
    lines = (base / raw_name).read_text(encoding="utf-8").splitlines()
    n = len(lines)
    home = np.full((n, len(home_ids), 2), np.nan)
    away = np.full((n, len(away_ids), 2), np.nan)
    ball = np.full((n, 2), np.nan)
    frame_ids = np.array([int(line[:line.index(":")]) for line in lines])
    if not np.array_equal(frame_ids, np.arange(1, n + 1)):
        raise ValueError("EPTS frames are not 1..n")
    for spec in root.iter("DataFormatSpecification"):
        first, last = int(spec.get("startFrame")) - 1, int(spec.get("endFrame")) - 1
        order = [channel_player[ref.get("playerChannelId")]
                 for ref in spec.iter("PlayerChannelRef") if ref.get("playerChannelId").endswith("_x")]
        rows = [lines[i].split(":") for i in range(first, last + 1)]
        players = pd.read_csv(io.StringIO("\n".join(r[1].replace(";", ",") for r in rows)),
                              header=None).to_numpy(dtype=np.float64).reshape(len(rows), len(order), 2)
        ball[first:last + 1] = pd.read_csv(io.StringIO("\n".join(r[2] for r in rows)),
                                           header=None).to_numpy(dtype=np.float64)
        for column, player in enumerate(order):
            if player_team[player] == 0:
                home[first:last + 1, home_ids.index(player)] = players[:, column]
            else:
                away[first:last + 1, away_ids.index(player)] = players[:, column]
    periods = []
    for period, key in ((1, "first_half"), (2, "second_half")):
        if params.get(f"{key}_start"):
            periods.append((period, int(params[f"{key}_start"]) - 1, int(params[f"{key}_end"]) - 1))
    raw = json.loads((base / events_name).read_text(encoding="utf-8"))["data"]
    events = []
    for event in raw:
        team = team_ids.index(event["team"]["id"]) if event.get("team") else None
        start = int(event["start"]["frame"]) - 1
        end = int(event["end"]["frame"]) - 1 if event.get("end") and event["end"].get("frame") else start
        events.append({"team": team, "type": event["type"]["name"].upper(),
                       "subtypes": _subtypes(event.get("subtypes")), "start": start,
                       "end": max(start, end), "period": int(event["period"])})
    return Game(f"metrica_{game}", FPS, PITCH, home, away, ball, periods, events)


def load(game, data_dir=None):
    missing = [name for name in FILES[game] if not (folder(game, data_dir) / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Metrica sample game {game} not found ({missing[0]}); "
                                "run `python -m validation --download` first")
    return _load_epts_game(game, data_dir) if game == 3 else _load_csv_game(game, data_dir)
