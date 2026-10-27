"""Rule-compliant offline validation on OTHER matches.

RULES.md forbids tuning or selecting variants on the provided match's ground truth (public window
included), but explicitly allows labelled data from other games. This package scores the
possession model on Metrica Sports' public sample games, whose possession and dead-ball labels are
derived mechanically from the games' own event data. Nothing here reads data of the provided match.
"""
