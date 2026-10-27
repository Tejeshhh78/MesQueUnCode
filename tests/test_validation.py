"""Synthetic tests for the other-match validation harness (no match data, no downloads)."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from possession.data import read_tracking
from validation import harness, labels
from validation.degrade import Degradation, degrade
from validation.metrica import Game


def event(team, kind, start, end=None, subtypes=(), period=1):
    return {"team": team, "type": kind, "subtypes": tuple(subtypes), "start": start,
            "end": start if end is None else end, "period": period}


def tiny_game(n=400, events=()):
    rng = np.random.default_rng(3)
    home = rng.uniform(0.2, 0.8, size=(n, 3, 2))
    away = rng.uniform(0.2, 0.8, size=(n, 3, 2))
    ball = np.column_stack([np.linspace(0.3, 0.7, n), np.full(n, 0.5)])
    return Game("tiny", 25, (105.0, 68.0), home, away, ball, [(1, 0, n - 1)], list(events))


class LabelTests(unittest.TestCase):
    def test_ownership_stoppage_and_restart(self):
        game = tiny_game(events=[
            event(0, "SET PIECE", 0, subtypes=("KICK OFF",)),
            event(0, "PASS", 0, 20),
            event(0, "BALL LOST", 60, 80, ("INTERCEPTION",)),
            event(1, "RECOVERY", 80),
            event(1, "PASS", 90, 120),
            event(1, "BALL OUT", 150, 170),
            event(0, "SET PIECE", 250, subtypes=("THROW IN",)),
            event(0, "PASS", 250, 270),
            event(1, "CHALLENGE", 300, subtypes=("GROUND", "WON")),
        ])
        codes, stats = labels.derive(game)
        np.testing.assert_array_equal(codes[:80], labels.H)
        np.testing.assert_array_equal(codes[80:170], labels.A)       # owner until the ball is out
        np.testing.assert_array_equal(codes[170:250], labels.D)      # dead until the restart
        np.testing.assert_array_equal(codes[250:], labels.H)         # a challenge does not transfer
        self.assertEqual(stats["stoppages"], 1)

    def test_advantage_after_foul_is_not_a_stoppage_and_slivers_are_absorbed(self):
        game = tiny_game(events=[
            event(0, "SET PIECE", 0, subtypes=("KICK OFF",)),
            event(1, "CHALLENGE", 50, subtypes=("TACKLE", "FAULT", "LOST")),
            event(0, "PASS", 60, 80),                                 # advantage played on
            event(1, "RECOVERY", 100),
            event(0, "RECOVERY", 102),                                # 2-frame sliver
            event(0, "FAULT RECEIVED", 200),
            event(0, "SET PIECE", 260, subtypes=("FREE KICK",)),
        ])
        codes, stats = labels.derive(game)
        np.testing.assert_array_equal(codes[:200], labels.H)          # sliver absorbed into H
        np.testing.assert_array_equal(codes[200:260], labels.D)
        self.assertEqual(stats["absorbed_short_spells"], 1)


class HarnessTests(unittest.TestCase):
    def test_features_match_the_solution_loader_on_the_same_positions(self):
        n = 30
        rng = np.random.default_rng(11)
        home, away = rng.uniform(0, 1, (n, 2, 2)), rng.uniform(0, 1, (n, 2, 2))
        ball = rng.uniform(0, 1, (n, 2))
        home[5:9, 0] = np.nan
        ball[12:15] = np.nan
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "meta.xml").write_text(
                '<root><Team id="h"><Name>Home</Name></Team><Team id="a"><Name>Away</Name></Team>'
                + '<Player teamId="h"/>' * 2 + '<Player teamId="a"/>' * 2 + "</root>")
            lines = []
            for t in range(n):
                groups = [f"{x},{y},1" for x, y in np.concatenate([home[t], away[t]])]
                lines.append(f"{t + 1}:" + ";".join(groups) + f":{ball[t, 0]},{ball[t, 1]},0\n")
            (root / "raw.txt").write_text("".join(lines).replace("nan", "NaN"))
            loaded = read_tracking(root / "meta.xml", root / "raw.txt", n, (105.0, 68.0))
        built = harness.build_features(ball, home, away, (105.0, 68.0), 25)
        np.testing.assert_allclose(built.distance, loaded.distance, rtol=1e-9)
        np.testing.assert_array_equal(built.player_count, loaded.player_count)
        np.testing.assert_allclose(built.ball[:, :2], loaded.ball[:, :2])

    def test_degradation_is_deterministic_and_hits_the_target_share(self):
        game = tiny_game(n=3000)
        codes = np.full(3000, labels.H, dtype=np.int8)
        codes[1000:1200] = labels.D
        settings = Degradation(alive_missing=0.4, mean_gap_s=0.5)
        first = degrade(game, codes, settings, seed=5)
        second = degrade(game, codes, settings, seed=5)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        alive = codes != labels.D
        share = np.mean(~np.isfinite(first[0][alive]).all(axis=1))
        self.assertAlmostEqual(share, 0.4, delta=0.03)


if __name__ == "__main__":
    unittest.main()
