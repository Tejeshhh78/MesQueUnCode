"""Mechanical H/A/D/X labels for OTHER games, derived from their event data.

The rules mirror the challenge definitions (CHALLENGE.md) as closely as event data allows:

* Dead ball runs from a stoppage trigger to the next restart. Triggers: BALL OUT (when the ball
  crosses the line, i.e. the event end), shots ending OUT or in a GOAL (event end), offside or
  end-of-half subtypes (event end), FAULT RECEIVED, fouls in challenges, and cards (event start).
  Restarts: every SET PIECE and every goal kick (recorded as PASS/BALL LOST with GOAL KICK).
  The trigger used is the earliest one after which no live on-ball action precedes the restart,
  so an advantage played after a foul does not create a phantom stoppage.
* While alive, the owner is the team of the latest on-ball action (set piece, pass, carry,
  recovery, shot, ball lost). Challenges and ball-out events never transfer ownership, so loose
  balls and duels stay with the last team in control, as the challenge rules require.
* Frames outside the periods are X; frames of a period before its first restart are D.

No person decides any label here: the event data belongs to other matches, which RULES.md allows.
"""

import numpy as np

H, A, D, X = 0, 1, 2, 3
ON_BALL = {"SET PIECE", "PASS", "CARRY", "RECOVERY", "SHOT", "BALL LOST"}
LIVE_ACTIONS = ON_BALL | {"CHALLENGE"}


def _is_restart(event):
    return event["type"] == "SET PIECE" or (
        event["type"] in ("PASS", "BALL LOST") and "GOAL KICK" in event["subtypes"])


def _trigger_frame(event):
    """Frame at which an event stops play, or None."""
    kind, subtypes = event["type"], set(event["subtypes"])
    if kind == "BALL OUT":
        return event["end"]
    if kind == "SHOT" and subtypes & {"OUT", "GOAL"}:
        return event["end"]
    if subtypes & {"OFFSIDE", "END HALF"}:
        return event["end"]
    if kind in ("FAULT RECEIVED", "CARD") or (kind == "CHALLENGE" and "FAULT" in subtypes):
        return event["start"]
    return None


def derive(game, n_frames=None, min_spell_frames=6):
    """Return (codes, stats) with codes in 0..3 over the game's frame index."""
    n = n_frames or len(game.ball_xy)
    codes = np.full(n, X, dtype=np.int8)
    stats = {"stoppages": 0, "fallback_triggers": 0}
    for period, first, last in game.periods:
        events = sorted((e for e in game.events if e["period"] == period),
                        key=lambda e: (e["start"], e["end"]))
        owner = np.full(last - first + 1, -1, dtype=np.int8)
        dead = np.zeros(last - first + 1, dtype=bool)
        # Ownership: forward-fill the team of the latest on-ball action. On the same frame the
        # losing side's BALL LOST is applied first and the winning side's RECOVERY last.
        priority = {"BALL LOST": 0, "RECOVERY": 2}
        marks = sorted(((e["start"], priority.get(e["type"], 1), e["team"]) for e in events
                        if e["type"] in ON_BALL and e["team"] is not None
                        and first <= e["start"] <= last), key=lambda m: (m[0], m[1]))
        marks = [(frame, team) for frame, _, team in marks]
        current = -1
        cursor = first
        for frame, team in marks + [(last + 1, None)]:
            owner[cursor - first:frame - first] = current
            cursor = max(cursor, frame)
            if team is not None:
                current = team
        # Dead intervals: trigger -> restart.
        restarts = [e for e in events if _is_restart(e)]
        previous = first
        for restart in restarts:
            stop = restart["start"]
            window = [e for e in events if previous <= e["start"] < stop or
                      (_trigger_frame(e) is not None and previous <= _trigger_frame(e) < stop)]
            live_starts = sorted(e["start"] for e in window if e["type"] in LIVE_ACTIONS
                                 and e is not restart and not _is_restart(e))
            triggers = sorted(t for t in (_trigger_frame(e) for e in window) if t is not None
                              and previous <= t < stop)
            begin = None
            for trigger in triggers:
                if not any(trigger < s < stop for s in live_starts):
                    begin = trigger
                    break
            if begin is None:
                # A restart without a recorded trigger (rare): stop at the last live action's end.
                ends = [e["end"] for e in window if e["type"] in LIVE_ACTIONS
                        and not _is_restart(e) and e["end"] < stop]
                begin = max(ends) if ends else previous
                if stop > previous:
                    stats["fallback_triggers"] += 1
            if stop > begin:
                dead[max(begin, first) - first:stop - first] = True
                stats["stoppages"] += 1
            previous = stop
        # Play stopped after the last restart (end of half without a restart).
        tail = [t for t in (_trigger_frame(e) for e in events) if t is not None and t >= previous]
        live_after = [e["start"] for e in events if e["type"] in LIVE_ACTIONS and e["start"] >= previous]
        for trigger in sorted(tail):
            if not any(s > trigger for s in live_after):
                dead[trigger - first:] = True
                break
        label = np.where(owner < 0, D, owner).astype(np.int8)
        label[dead] = D
        stats["absorbed_short_spells"] = stats.get("absorbed_short_spells", 0) + _absorb_short(
            label, min_spell_frames)
        codes[first:last + 1] = label
    return codes, stats


def _absorb_short(label, min_frames):
    """Give spells shorter than min_frames to the preceding (else following) state.

    EVALUATION.md reports that a uniform one-frame lag costs exactly nothing on the full-match
    ground truth; at IoU 0.7 that is only possible if no possession is shorter than about six
    frames. Event timestamps produce a few such slivers (a recovery and a loss logged on almost
    the same frame), so they are absorbed to keep the validation labels structurally comparable.
    """
    absorbed = 0
    changed = True
    while changed and min_frames > 1:
        changed = False
        breaks = np.flatnonzero(label[1:] != label[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(label)]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a < min_frames and label[a] in (H, A) and (a > 0 or b < len(label)):
                label[a:b] = label[a - 1] if a > 0 else label[b]
                absorbed += 1
                changed = True
                break
    return absorbed
