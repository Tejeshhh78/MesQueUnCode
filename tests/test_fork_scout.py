"""Offline regression checks for fork discovery and source evidence."""
import json
import unittest
from unittest.mock import patch

from tools import fork_scout as scout


class ForkScoutTests(unittest.TestCase):
    def test_paginated_nested_forks_visited_once(self):
        first_page = "https://api.github.com/repos/root/test/forks?per_page=100&sort=oldest"
        next_page = "https://api.github.com/repos/root/test/forks?page=2"
        child_page = first_page.replace("root/test", "b/test")
        responses = {
            first_page: ([{"full_name": "b/test", "forks_count": 1}], {"Link": f'<{next_page}>; rel="next"'}),
            next_page: ([{"full_name": "c/test", "forks_count": 0}], {}),
            child_page: ([{"full_name": "d/test", "forks_count": 0},
                          {"full_name": "c/test", "forks_count": 0}], {}),
        }

        def fetch(url):
            data, headers = responses.pop(url)
            return json.dumps(data).encode(), headers

        with patch.object(scout, "github", side_effect=fetch):
            self.assertEqual(scout.forks("root/test"), (["b/test", "c/test", "d/test"], []))
        self.assertFalse(responses)

    def test_source_hints_ignore_starter_lines_and_preserve_permalinks(self):
        data = b"# viterbi baseline\n# kalman candidate\n"
        with patch.object(scout, "github", return_value=(data, {})):
            hits = scout.evidence("owner/repo", "a" * 40, "src/team file.py", {"# viterbi baseline"})["hits"]
        self.assertEqual(hits, {"Balllücken": "https://github.com/owner/repo/blob/" + "a" * 40
                               + "/src/team%20file.py#L2"})

    def test_partial_discovery_reports_error_without_losing_earlier_forks(self):
        page = ([{"full_name": "b/test", "forks_count": 1}], {})
        with patch.object(scout, "github", side_effect=[(json.dumps(page[0]).encode(), page[1]),
                                                        OSError("rate limit")]):
            names, errors = scout.forks("root/test")
        self.assertEqual(names, ["b/test"])
        self.assertEqual(errors, ["b/test: rate limit"])
