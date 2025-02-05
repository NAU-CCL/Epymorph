# ruff: noqa: PT009,PT027
import unittest

from epymorph.geography.us_census import StateScope


class StateScopeTest(unittest.TestCase):
    def test_state_scope_in_states(self):
        scope = StateScope.in_states(["04", "35", "08"], year=2020)
        self.assertTrue(scope.granularity, "state")
        self.assertTrue(scope.includes, ["04", "08", "35"])
        self.assertTrue(scope.includes_granularity, "state")

    def test_state_scope_in_states_by_code(self):
        scope = StateScope.in_states(["AZ", "NM", "CO"], year=2020)
        self.assertTrue(scope.granularity, "state")
        self.assertTrue(scope.includes, ["04", "08", "35"])
        self.assertTrue(scope.includes_granularity, "state")
