"""Paths and every tunable number of the pipeline, with the reason each one has its value.

How the values were chosen -- and what is never used
-----------------------------------------------------
The organisers' rule clarification: anything trained, fitted, calibrated or tuned must use
external data only -- never the ground truth and never the provided match.

* Values that are *fitted* (the motion threshold and the team-shape model) come from
  ``solution/calibration.json``, written by ``solution/calibrate.py`` from Metrica Sports' public
  sample games -- other matches, with labels derived from their own event data. At run time on the
  provided match they are only applied, never refitted.
* Values that are *tuned* (control_sigma_m, switch_cost, min_spell_seconds) were chosen by
  ``solution/tune_external.py`` on the same external games, and are marked as such below.
* Every other value is a fixed design choice from football and physics knowledge (pitch geometry,
  pass and dribble speeds, line widths, frame rate), stated next to it. None was set by measuring
  the provided match or by scoring against any part of its ground truth.
* The public 5-minute ground truth is used only for automatic correctness checks of the pipeline.

Units: distances in metres, speeds in m/s, durations in frames at 25 fps unless the name says
seconds. Costs are in nats (negative log-likelihood) and are what the Viterbi decoder adds up.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from typing import Dict, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.environ.get("MQUC_CACHE_DIR") or os.path.join(ROOT, ".cache", "mesqueuncode")
RESULTS_DIR = os.path.join(ROOT, "solution", "results")
DEFAULT_OUTPUT = os.path.join(RESULTS_DIR, "mes_que_un_code.csv")
FPS = 25
CALIBRATION_PATH = os.path.join(ROOT, "solution", "calibration.json")


def _load_calibration() -> dict:
    """The external-data calibration. Only calibrate.py may run without it (it writes it)."""
    if os.path.isfile(CALIBRATION_PATH):
        with open(CALIBRATION_PATH, encoding="utf-8") as handle:
            return json.load(handle)
    if os.environ.get("MQUC_CALIBRATING") == "1":
        return {"motion": {"still_speed": 1.4, "still_scale": 0.3},
                "shape": {"mean": [], "std": [], "weights": []}}
    raise FileNotFoundError(
        "{} is missing. It is committed with the code; regenerate it from the external sample "
        "games with:  python solution/calibrate.py --download --write".format(CALIBRATION_PATH))


CALIBRATION = _load_calibration()

# Smart Tagging codes whose Team tag names the side IN POSSESSION, chosen by what the codes mean:
# a build-up, a progression and an attacking transition are, by definition, phases of the team
# with the ball. The "DEFENDING IN ... THIRD" and "DEFENSIVE TRANSITION" codes name the team
# without it and describe the same phases from the other side, so using them too would count each
# phase twice. Codes about locations ("BALL IN THE BOX", "PLAYERS IN THE BOX") are not used.
POSSESSION_TAG_CODES = ("BUILD UP", "PROGRESSION", "ATTACKING TRANSITION")

# Set-piece Type values (lower case) that open with a dead ball. Kick-offs are handled separately
# because their tag is a short window centred on the restart rather than starting at the stoppage.
DEAD_START_SET_PIECES = ("throw-in", "corner kick", "goal kick", "free kick")
KICKOFF_TYPES = ("kick off (start)", "kick off (after goal)")


@dataclass(frozen=True)
class Params:
    # -- ball track -----------------------------------------------------------------------------
    # Linearly bridge ball gaps up to this long (0.4 s). Longer gaps are real losses: interpolating
    # across them invents a trajectory through exactly the crowded moments the tracker lost.
    max_interp_gap: int = 10
    # Ball speed is estimated over +/- this many frames to keep detection jitter (a few cm per
    # frame, i.e. ~1-2 m/s of fake speed) from looking like a touch.
    speed_halfwidth: int = 2

    # -- team control evidence --------------------------------------------------------------------
    # Scale of the Gaussian "could be playing the ball" kernel on the nearest player of each team.
    # TUNED ON EXTERNAL GAMES by tune_external.py (grid 0.6/0.8/1.0/1.5 m on Metrica's sample
    # games). 1.0 m -- about a stride -- won; it agrees with the football reading that a player
    # controls the ball within reach, not a few metres away.
    control_sigma_m: float = 1.0
    # Above this ball speed the ball is a pass or a shot, not a controlled ball: a player it flies
    # past has not won it. Real passes run 10-25 m/s; dribbles stay under ~6 m/s.
    flight_speed: float = 9.0
    # Evidence multiplier for a ball in flight (between 0 and 1).
    flight_weight: float = 0.15
    # Log-odds per frame of a clear, slow, close control by one team. Frames are strongly
    # autocorrelated, so this is far smaller than an independent-frame likelihood would be.
    team_evidence: float = 0.35
    # Half-width (frames) of the centred mean filter over the team evidence. Removes the flicker
    # of two players a few cm apart swapping "nearest". Overridden by --window.
    evidence_window: int = 6
    # Inside a ball gap longer than max_interp_gap, the team evidence is a weak blend of the last
    # evidence before the gap and the first after it, scaled by this. It only breaks ties so a
    # hand-over hidden in a gap is placed mid-gap rather than at either edge.
    gap_blend: float = 0.05

    # -- Smart Tagging team prior ---------------------------------------------------------------
    # Log-odds per frame for the possession team named by a POSSESSION_TAG_CODES interval. Weak
    # on purpose: phases are not possessions, and the tagging is automatic.
    tag_team_evidence: float = 0.03

    # -- team-shape prior (fitted on external games) -------------------------------------------------
    # Where the ball is not seen, a logistic regression on collective team shape -- width, length,
    # territory, forward movement -- says which side looks like it has the ball. Its feature
    # standardisation and weights are fitted by calibrate.py on Metrica's public sample games with
    # event-derived possession labels, and only applied here. Log-odds per frame =
    # shape_evidence * logit, on frames with no ball position only.
    shape_evidence: float = 0.02
    shape_halfwidth: int = 12
    shape_min_players: int = 7
    shape_clip: float = 4.0
    shape_mean: Tuple[float, ...] = tuple(CALIBRATION["shape"]["mean"])
    shape_std: Tuple[float, ...] = tuple(CALIBRATION["shape"]["std"])
    shape_weights: Tuple[float, ...] = tuple(CALIBRATION["shape"]["weights"])

    # -- homography -----------------------------------------------------------------------------
    # Where the tactical camera-to-pitch fit misses by more than this many pixels, team evidence is
    # scaled by homography_weight. DATA.md documents a median of 0.24 px during play and ~1.5 px
    # when the camera is off the pitch; 5 px is well clear of both.
    homography_max_error: float = 5.0
    homography_weight: float = 0.5

    # -- dead ball evidence -----------------------------------------------------------------------
    # The ball is out when it is this far past a line. Lines are 12 cm wide and the tracking is
    # noisy, so a margin avoids calling a ball on the touchline dead.
    out_margin_m: float = 0.75
    # Log-odds per frame in favour of D while the ball is out of the pitch.
    out_evidence: float = 0.6
    # Mean player speed around which play looks stopped, and how sharply: a logistic regression of
    # dead ball on smoothed mean player speed, fitted by calibrate.py on Metrica's public sample
    # games (event-derived dead-ball labels, tracking degraded towards the automatic ATD).
    still_speed: float = CALIBRATION["motion"]["still_speed"]
    still_scale: float = CALIBRATION["motion"]["still_scale"]
    # Mean player speed is smoothed over +/- this many frames (1 s) before use.
    motion_halfwidth: int = 25
    # Weight of the motion evidence (log-odds per frame at the extremes). Motion smoothed over
    # +/- 1 s carries about one independent observation per second, so a per-frame weight of a few
    # hundredths keeps a single slow second from outweighing the decoder's stoppage costs.
    motion_evidence: float = 0.08
    # Fewer tracked players than this (out of ~22) usually means the camera is not on play; no
    # motion evidence is taken from such frames.
    min_players_for_motion: int = 12
    # A ball that has barely moved for this long (seconds), with players slow, is being placed for
    # a restart.
    stationary_speed: float = 0.6
    stationary_seconds: float = 1.5
    stationary_evidence: float = 0.25

    # A ball travelling at pass speed inside the pitch is in play: during stoppages it is carried,
    # rolled or tossed to the taker. Evidence against D.
    moving_ball_speed: float = 4.0
    moving_ball_evidence: float = 0.1

    # Smart Tagging set pieces that open with a stoppage: D prior over the first seconds of the
    # 35 s window. Its value per frame is small; it tips decisions, it does not make them.
    set_piece_dead_seconds: float = 4.0
    set_piece_dead_evidence: float = 0.15

    # -- kick-offs --------------------------------------------------------------------------------
    # The ball must be within this distance of the centre spot to be a kick-off ball.
    kickoff_radius_m: float = 3.0
    # Ball speed that marks the kick.
    kickoff_speed: float = 2.0

    # -- decoder (all costs in nats) -------------------------------------------------------------
    # Cost of a hand-over H<->A. Divided by team_evidence it is roughly the number of frames of
    # clear control the new team needs before the decoder believes the change: 1.0 / 0.35 ~ 3
    # frames of unambiguous on-the-ball evidence, more when it is mixed, and the 1 s minimum
    # possession below still applies. TUNED ON EXTERNAL GAMES by tune_external.py (grid
    # 0.25-2.0): the hand-picked 4.0 kept too few, too long possessions (0.48x the true count).
    switch_cost: float = 1.0
    # Cost of the ball going dead, and of play restarting.
    dead_enter_cost: float = 3.0
    dead_exit_cost: float = 3.0
    # Fixed per-frame bias towards play being alive. Live play is the majority of graded time.
    alive_bias: float = 0.02

    # -- spell clean-up -------------------------------------------------------------------------
    # Possession spells shorter than this (seconds) are absorbed by their neighbours. A possession
    # needs "a few seconds" of control (CHALLENGE.md); 1 s is a conservative floor below that.
    # Checked on external games by tune_external.py (0.7 vs 1.0 s: 1.0 kept).
    min_spell_seconds: float = 1.0
    # Dead-ball runs shorter than this inside one team's possession are bridged.
    min_dead_seconds: float = 1.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def describe(self) -> Dict[str, object]:
        """to_dict() without the fitted shape arrays, for printing."""
        out = asdict(self)
        for key in ("shape_mean", "shape_std", "shape_weights"):
            out[key] = "<{} values from calibration.json>".format(len(out[key]))
        return out

    def with_overrides(self, **overrides: float) -> "Params":
        unknown = sorted(set(overrides) - set(asdict(self)))
        if unknown:
            raise KeyError("unknown parameter(s): {}".format(", ".join(unknown)))
        return replace(self, **overrides)


DEFAULT_PARAMS = Params()
