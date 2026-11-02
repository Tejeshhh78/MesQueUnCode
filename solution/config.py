"""Paths and every tunable number of the pipeline, with the reason each one has its value.

How the defaults were chosen
----------------------------
From football and physics priors and from *unlabelled* statistics of the provided tracking
(distributions of ball speed, nearest-player distance, mean player speed, Smart Tagging intervals).
None of them was fitted to the public 5-minute ground truth: RULES.md forbids it, and
EVALUATION.md shows that tuning on 17 possessions picks the wrong answer even for a one-parameter
baseline. The public window is used only to check that the pipeline is correct.

Units: distances in metres, speeds in m/s, durations in frames at 25 fps unless the name says
seconds. Costs are in nats (negative log-likelihood) and are what the Viterbi decoder adds up.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, replace
from typing import Dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.environ.get("MQUC_CACHE_DIR") or os.path.join(ROOT, ".cache", "mesqueuncode")
RESULTS_DIR = os.path.join(ROOT, "solution", "results")
DEFAULT_OUTPUT = os.path.join(RESULTS_DIR, "mes_que_un_code.csv")
FPS = 25

# Smart Tagging codes whose Team tag names the side IN POSSESSION. Checked without ground truth:
# inside these intervals the nearest player to a visible ball belongs to the tagged team 71-87% of
# the time. The DEFENDING ... codes are the same intervals with the defending team named (their
# agreement is the mirror image, 13-38%), so using them as well would count each phase twice.
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
    # Median nearest-player distance during play is ~1.9 m, so 1.5 m separates on-the-ball from
    # nearby-but-not-touching.
    control_sigma_m: float = 1.5
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

    # -- team-shape prior, self-supervised -----------------------------------------------------------
    # Where the ball is not seen (41% of graded frames), a logistic regression on collective team
    # shape -- width, length, territory, forward movement -- says which side looks like it has the
    # ball. It is fitted on this match's *silver* labels: frames where a rule, not a person, finds
    # one team's player within silver_near_m of a slow ball and nobody from the other team within
    # silver_far_m. Trained on one half and tested on the other it agrees with held-out silver
    # labels ~84% of the time (majority class: ~62%). Log-odds per frame = shape_evidence * logit.
    shape_evidence: float = 0.02
    shape_halfwidth: int = 12
    shape_min_players: int = 7
    shape_l2: float = 1.0
    shape_clip: float = 4.0
    shape_min_training_frames: int = 500
    silver_near_m: float = 1.0
    silver_far_m: float = 2.5
    silver_max_speed: float = 6.0

    # -- homography -----------------------------------------------------------------------------
    # Where the tactical camera-to-pitch fit misses by more than this many pixels (median during
    # play is 0.24, ~1.5 across halftime), team evidence is scaled by homography_weight.
    homography_max_error: float = 5.0
    homography_weight: float = 0.5

    # -- dead ball evidence -----------------------------------------------------------------------
    # The ball is out when it is this far past a line. Lines are 12 cm wide and the tracking is
    # noisy, so a margin avoids calling a ball on the touchline dead.
    out_margin_m: float = 0.75
    # Log-odds per frame in favour of D while the ball is out of the pitch.
    out_evidence: float = 0.6
    # Mean player speed around which play looks stopped, and how sharply. Calibrated without ground
    # truth on algorithmic silver labels (see docs/MODEL.md): frames where the ball is visibly out
    # of the pitch (dead) against frames where an in-pitch ball moves at pass speed (live). The
    # likelihood ratio of the two crosses zero between 1.5 and 1.75 m/s, and both silver sets are
    # biased towards faster players than their class average, so the midpoint is taken slightly low.
    still_speed: float = 1.4
    still_scale: float = 0.3
    # Mean player speed is smoothed over +/- this many frames (1 s) before use.
    motion_halfwidth: int = 25
    # Weight of the motion evidence (log-odds per frame at the extremes). The silver-label ratio
    # reaches ~2 nats per frame, but motion smoothed over 2 s is one independent sample every
    # ~50 frames, so its per-frame weight is ~2 / 25.
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
    # clear control the new team needs before the decoder believes the change: 4.0 / 0.35 ~ 11
    # frames (~0.5 s) of unambiguous on-the-ball evidence, more when it is mixed. This is the
    # hysteresis that turns a deflection or a lost aerial duel into "no change".
    switch_cost: float = 4.0
    # Cost of the ball going dead, and of play restarting.
    dead_enter_cost: float = 3.0
    dead_exit_cost: float = 3.0
    # Fixed per-frame bias towards play being alive. Live play is the majority of graded time.
    alive_bias: float = 0.02

    # -- spell clean-up -------------------------------------------------------------------------
    # Possession spells shorter than this (seconds) are absorbed by their neighbours. A possession
    # needs "a few seconds" of control (CHALLENGE.md); 1 s is a conservative floor below that.
    min_spell_seconds: float = 1.0
    # Dead-ball runs shorter than this inside one team's possession are bridged.
    min_dead_seconds: float = 1.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def with_overrides(self, **overrides: float) -> "Params":
        unknown = sorted(set(overrides) - set(asdict(self)))
        if unknown:
            raise KeyError("unknown parameter(s): {}".format(", ".join(unknown)))
        return replace(self, **overrides)


DEFAULT_PARAMS = Params()
