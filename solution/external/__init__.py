"""External (other-match) data used to calibrate and validate the model.

``metrica.py``, ``labels.py`` and ``degrade.py`` are taken from Leo Hofmann's validation harness
(branch ``dev_LH``, commit 166c681) with only import paths adjusted. They read Metrica Sports'
public sample games (https://github.com/metrica-sports/sample-data, © Metrica Sports, shared for
the football analytics community with acknowledgement), derive H/A/D labels mechanically from
those games' own events, and degrade the clean tracking towards the competition's automatic ATD.

Nothing in this package reads the provided BAR-ATH match.
"""
