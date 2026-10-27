"""Synthetic Smart-Tagging-like phases for OTHER games, with an explicit error model.

The sample games have no Smart Tagging, so this produces phase intervals in the format of
possession.data.read_events from the derived labels, then corrupts them: late starts, loose ends,
missed phases, wrong teams and N/A teams. It only exists to check that conclusions drawn without
tags survive their presence; the error model is an assumption, not a measurement of Metrica's tagger.
"""

from dataclasses import dataclass

import numpy as np

from evaluation import possession_metrics as pm

ATTACK = ("buildup", "progression", "creatingchances", "ballinfinalthird", "ballinthebox",
          "playersinthebox", "longball")
DEFENCE = ("defendinginattackingthird", "defendinginmiddlethird", "defendingindefensivethird")


@dataclass(frozen=True)
class TagModel:
    attack_rate: float = 0.6      # share of possessions >= 4 s that get an attacking phase
    defence_rate: float = 0.4     # ... and a defending phase for the opponent
    transition_rate: float = 0.3  # share of open-play turnovers tagged as transitions
    setpiece_rate: float = 0.55   # share of stoppages tagged as SET PIECES
    wrong_team: float = 0.1       # phase credited to the wrong team
    start_delay_s: tuple = (0.5, 2.5)
    end_jitter_s: float = 1.5


def synthesise(codes, runs, fps=25, model=TagModel(), seed=0):
    rng = np.random.default_rng(seed)
    names = ("Home", "Away")
    events = []

    def team_name(side):
        if side is None:
            return "N/A"
        return names[1 - side] if rng.random() < model.wrong_team else names[side]

    def add(start, stop, code, side):
        start, stop = int(max(0, start)), int(min(len(codes), stop))
        if stop - start >= fps // 2:
            events.append({"start": start, "stop": stop, "code": code, "tags": {"team": team_name(side)}})

    segments = pm.segments(codes, runs)
    for index, (a, b, k) in enumerate(segments):
        if k in (0, 1) and b - a >= 4 * fps:
            begin = a + rng.uniform(*model.start_delay_s) * fps
            end = b + rng.normal(0, model.end_jitter_s) * fps
            if rng.random() < model.attack_rate:
                add(begin, max(begin + fps, end), str(rng.choice(ATTACK)), k)
            if rng.random() < model.defence_rate:
                add(begin, max(begin + fps, end), str(rng.choice(DEFENCE)), 1 - k)
        if k in (0, 1) and index and segments[index - 1][2] == 1 - k and segments[index - 1][1] == a:
            if rng.random() < model.transition_rate:
                begin = a + rng.uniform(0, 1) * fps
                length = rng.uniform(3, 6) * fps
                add(begin, begin + length, "attackingtransition", k)
                add(begin, begin + length, "defensivetransition", 1 - k)
        if k == 2 and index + 1 < len(segments) and rng.random() < model.setpiece_rate:
            restart_team = segments[index + 1][2] if segments[index + 1][2] in (0, 1) else None
            throw_in = rng.random() < 0.4
            add(b - rng.uniform(3, 10) * fps, b + rng.uniform(1, 4) * fps, "setpieces",
                None if throw_in else restart_team)
    return events
