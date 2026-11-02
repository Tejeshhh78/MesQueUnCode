"""Turn per-frame evidence into possession spells with a three-state Viterbi decoder.

Why a decoder and not a per-frame classifier
--------------------------------------------
spellF1 scores possessions as whole intervals, so one flicker inside a long possession splits it in
two and costs a true positive *and* adds a false one. A per-frame classifier has no notion of that
cost; a decoder with an explicit price on every change of state does. Viterbi finds, per half, the
single label sequence that maximises

    sum over frames of score[state]  -  sum over changes of transition cost

which is the best segmentation under a per-change penalty. That is hysteresis done exactly: a
hand-over happens only when the new team's accumulated evidence pays for it, so a deflection or a
lost aerial duel does not register, and a run of zero evidence (ball not seen) carries the current
state through untouched -- the CHALLENGE.md rule that a loose ball stays with the last team.

States are H, A, D. A change through D (H -> D -> A, a throw-in to the other side) is priced as
leaving and re-entering play, not as a hand-over.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from solution.config import Params
from solution.features import Evidence

H, A, D = 0, 1, 2
STATE_LABELS = np.array(["H", "A", "D"])
FORBIDDEN = 1e9


def emission_scores(evidence: Evidence, params: Params) -> np.ndarray:
    """(n, 3) per-frame scores (higher is likelier) for H, A, D.

    The team log-odds is split symmetrically between H and A, and the dead log-odds between D and
    the two live states, so each one-unit change of evidence is a one-unit change of the margin
    the decoder sees.

    D also receives |team| / 2, the score of the better team. A player standing next to the ball
    says *whose* ball it would be, not whether it is in play -- the taker holds it for every
    throw-in -- so without this a dead ball in someone's hands would be pulled back to live play.
    Alive versus dead is decided by the dead evidence alone.
    """
    team, dead = evidence.team, evidence.dead
    scores = np.empty((len(team), 3))
    scores[:, H] = 0.5 * team - 0.5 * dead + params.alive_bias
    scores[:, A] = -0.5 * team - 0.5 * dead + params.alive_bias
    scores[:, D] = 0.5 * dead + 0.5 * np.abs(team)
    scores[evidence.forced_dead, H] = -FORBIDDEN
    scores[evidence.forced_dead, A] = -FORBIDDEN
    return scores


def transition_costs(params: Params) -> np.ndarray:
    """3x3 cost of moving from state i (row) to state j (column)."""
    cost = np.zeros((3, 3))
    cost[H, A] = cost[A, H] = params.switch_cost
    cost[H, D] = cost[A, D] = params.dead_enter_cost
    cost[D, H] = cost[D, A] = params.dead_exit_cost
    return cost


def viterbi(scores: np.ndarray, cost: np.ndarray, start_state: int = D) -> np.ndarray:
    """Most likely state path for one contiguous span. Deterministic: ties keep the current state.

    The first frame is free to be any state, but start_state gets a head start equal to the cheapest
    way of entering any other state from it, so a half begins dead unless the evidence says
    otherwise (it opens with a kick-off).
    """
    n = len(scores)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    pointers = np.zeros((n, 3), dtype=np.int8)
    total = [scores[0, s] - (0.0 if s == start_state else cost[start_state, s]) for s in range(3)]
    trans = cost.tolist()
    rows = scores.tolist()
    for f in range(1, n):
        row = rows[f]
        new_total = [0.0, 0.0, 0.0]
        back = pointers[f]
        for j in range(3):
            # Prefer staying in j on ties, then the lower state index -- fixed order, so the
            # result never depends on floating-point accidents of iteration.
            best_i, best = j, total[j] - trans[j][j]
            for i in range(3):
                candidate = total[i] - trans[i][j]
                if candidate > best:
                    best_i, best = i, candidate
            new_total[j] = best + row[j]
            back[j] = best_i
        total = new_total
    path = np.empty(n, dtype=np.int8)
    state = int(np.argmax(total))
    for f in range(n - 1, -1, -1):
        path[f] = state
        state = int(pointers[f, state])
    return path


def decode(evidence: Evidence, params: Params,
           halves: Sequence[Tuple[int, int, int]]) -> np.ndarray:
    """State per frame for every graded half; -1 on frames outside the halves."""
    scores = emission_scores(evidence, params)
    cost = transition_costs(params)
    states = np.full(len(scores), -1, dtype=np.int8)
    for _, first, last in halves:
        states[first:last + 1] = viterbi(scores[first:last + 1], cost, start_state=D)
    return states
