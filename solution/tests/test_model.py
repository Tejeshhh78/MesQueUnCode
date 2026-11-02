"""Decoding, spell clean-up, halftime handling, submission format and determinism."""

import contextlib
import hashlib
import io
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from solution import config, loaders, possession_model, postprocess  # noqa: E402
from solution.possession_model import A, D, H  # noqa: E402
from solution.run_solution import main, parse_overrides, predict  # noqa: E402
from solution.tests import helpers  # noqa: E402

PARAMS = config.DEFAULT_PARAMS


def decode_scores(team, dead=None, switch=4.0):
    """Run the Viterbi on hand-written evidence."""
    team = np.asarray(team, dtype=float)
    dead = np.zeros_like(team) if dead is None else np.asarray(dead, dtype=float)
    scores = np.column_stack([0.5 * team - 0.5 * dead, -0.5 * team - 0.5 * dead, 0.5 * dead])
    params = PARAMS.with_overrides(switch_cost=switch)
    return possession_model.viterbi(scores, possession_model.transition_costs(params),
                                    start_state=H)


class ViterbiTest(unittest.TestCase):
    def test_a_short_burst_of_opponent_evidence_does_not_switch(self):
        team = [0.5] * 50 + [-0.5] * 5 + [0.5] * 50   # a deflection: 5 frames
        np.testing.assert_array_equal(decode_scores(team), H)

    def test_sustained_opponent_evidence_does_switch(self):
        team = [0.5] * 50 + [-0.5] * 60
        path = decode_scores(team)
        self.assertEqual(path[0], H)
        self.assertEqual(path[-1], A)
        self.assertEqual(int(np.sum(path[1:] != path[:-1])), 1)
        self.assertEqual(int(np.flatnonzero(path == A)[0]), 50)  # placed where evidence turns

    def test_no_evidence_carries_the_current_team(self):
        team = [0.5] * 30 + [0.0] * 200
        np.testing.assert_array_equal(decode_scores(team), H)

    def test_dead_evidence_produces_a_stoppage(self):
        team = [0.5] * 50 + [0.0] * 50 + [-0.5] * 50
        dead = [0.0] * 50 + [1.0] * 50 + [0.0] * 50
        path = decode_scores(team, dead)
        self.assertTrue(np.all(path[55:95] == D))
        self.assertEqual(path[-1], A)

    def test_forced_dead_frames_are_always_d(self):
        match = helpers.possession_script()
        from solution import features
        evidence = features.build_evidence(match, PARAMS, helpers.PITCH)
        evidence.forced_dead[50:120] = True
        states = possession_model.decode(evidence, PARAMS, match.halves)
        self.assertTrue(np.all(states[50:120] == D))


class CleanupTest(unittest.TestCase):
    def test_short_spell_inside_a_possession_is_absorbed(self):
        states = np.array([H] * 40 + [A] * 5 + [H] * 40, dtype=np.int8)
        out = postprocess.clean_span(states, min_spell=25, min_dead=0)
        np.testing.assert_array_equal(out, H)

    def test_short_spell_next_to_a_stoppage_joins_the_other_team(self):
        states = np.array([A] * 40 + [H] * 5 + [D] * 40, dtype=np.int8)
        out = postprocess.clean_span(states, min_spell=25, min_dead=0)
        np.testing.assert_array_equal(out[:45], A)
        np.testing.assert_array_equal(out[45:], D)

    def test_short_spell_between_stoppages_is_dead(self):
        states = np.array([D] * 40 + [H] * 5 + [D] * 40, dtype=np.int8)
        out = postprocess.clean_span(states, min_spell=25, min_dead=0)
        np.testing.assert_array_equal(out, D)

    def test_short_dead_blip_inside_one_team_is_bridged(self):
        states = np.array([H] * 40 + [D] * 5 + [H] * 40, dtype=np.int8)
        out = postprocess.clean_span(states, min_spell=0, min_dead=25)
        np.testing.assert_array_equal(out, H)

    def test_short_dead_between_different_teams_is_kept(self):
        states = np.array([H] * 40 + [D] * 5 + [A] * 40, dtype=np.int8)
        out = postprocess.clean_span(states, min_spell=0, min_dead=25)
        np.testing.assert_array_equal(out, states)

    def test_long_spells_are_untouched(self):
        states = np.array([H] * 40 + [A] * 40 + [D] * 40, dtype=np.int8)
        out = postprocess.clean_span(states, min_spell=25, min_dead=25)
        np.testing.assert_array_equal(out, states)


class HalftimeAndEdgesTest(unittest.TestCase):
    def setUp(self):
        self.match = helpers.possession_script()
        self.labels = predict(self.match, PARAMS)

    def test_ungraded_frames_are_x_and_graded_never_are(self):
        graded = loaders.graded_mask(helpers.N_FRAMES, self.match.halves)
        self.assertTrue(np.all(self.labels[~graded] == "X"))
        self.assertFalse(np.any(self.labels[graded] == "X"))
        self.assertTrue(set(np.unique(self.labels)) <= {"H", "A", "D", "X"})

    def test_first_and_last_graded_frames_are_labelled(self):
        for _, first, last in self.match.halves:
            self.assertIn(self.labels[first], ("H", "A", "D"))
            self.assertIn(self.labels[last], ("H", "A", "D"))

    def test_the_scripted_story_is_recovered(self):
        self.assertTrue(np.all(self.labels[60:190] == "H"))
        self.assertTrue(np.all(self.labels[260:440] == "A"))
        self.assertTrue(np.all(self.labels[610:790] == "H"))
        self.assertTrue(np.all(self.labels[810:890] == "D"))
        self.assertTrue(np.all(self.labels[910:990] == "A"))

    def test_no_spell_crosses_halftime(self):
        stats = postprocess.spell_stats(self.labels, self.match.halves, 25)
        # H, A in the first half; H and A either side of the stoppage in the second.
        self.assertEqual(stats["possessions"], 4)
        self.assertEqual(stats["stoppages"], 1)


class SubmissionFormatTest(unittest.TestCase):
    def test_written_file_passes_the_official_validation(self):
        labels = np.full(199319, "X", dtype="<U1")
        labels[11185:90652] = "H"
        labels[110808:190198] = "D"
        with tempfile.TemporaryDirectory() as folder:
            path = postprocess.write_submission(os.path.join(folder, "s.csv"), labels)
            table = pm.read_label_csv(path)
            self.assertIsNone(pm.validate_submission(table, 199319))
            np.testing.assert_array_equal(table["frame"].values, np.arange(199319))
            with open(path) as handle:
                self.assertEqual(handle.readline(), "frame,label\n")

    def test_writing_is_byte_deterministic(self):
        labels = predict(helpers.possession_script(), PARAMS)
        with tempfile.TemporaryDirectory() as folder:
            first = postprocess.write_submission(os.path.join(folder, "a.csv"), labels)
            second = postprocess.write_submission(os.path.join(folder, "b.csv"),
                                                  predict(helpers.possession_script(), PARAMS))
            digests = []
            for path in (first, second):
                with open(path, "rb") as handle:
                    digests.append(hashlib.sha256(handle.read()).hexdigest())
        self.assertEqual(digests[0], digests[1])


class CommandLineTest(unittest.TestCase):
    def test_overrides_are_typed_and_checked(self):
        out = parse_overrides(["switch_cost=6", "evidence_window=10"])
        self.assertEqual(out, {"switch_cost": 6.0, "evidence_window": 10})
        self.assertIsInstance(out["evidence_window"], int)
        with self.assertRaises(KeyError):
            parse_overrides(["no_such_thing=1"])
        with self.assertRaises(ValueError):
            parse_overrides(["switch_cost"])

    def test_missing_data_fails_clearly(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(loaders.DataUnavailableError) as caught:
                loaders.check_data_available(data_dir=folder)
            self.assertIn("MTAH_DATA_DIR", str(caught.exception))
            self.assertIn("run_evaluation.py", str(caught.exception))
            with contextlib.redirect_stderr(io.StringIO()) as stderr:
                code = main(["--data-dir", folder, "--output", os.path.join(folder, "out.csv")])
            self.assertEqual(code, 2)
            self.assertIn("match data not found", stderr.getvalue())
            self.assertFalse(os.path.exists(os.path.join(folder, "out.csv")))


if __name__ == "__main__":
    unittest.main()
