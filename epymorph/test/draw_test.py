import unittest

from sympy import Max, symbols

import epymorph.draw
from epymorph import ipm_library


class DrawTest(unittest.TestCase):
    def test_build_edges(self):
        """
        Tests the conversion between ipm edges and an edge tracker object
        WARNING: Will stop working if SIRS model changes and/or is deleted!
        """
        test_ipm = ipm_library["sirs"]()
        test_edge_set = epymorph.draw.build_ipm_edge_set(test_ipm)

        S, I, R, beta, gamma, xi = symbols("S I R beta gamma xi")

        N = Max(1, S + I + R)

        self.assertEqual(test_edge_set.edge_dict[("S", "I")], beta * S * I / N)
        self.assertEqual(test_edge_set.edge_dict[("I", "R")], gamma * I)
        self.assertEqual(test_edge_set.edge_dict[("R", "S")], xi * R)
        self.assertEqual(len(test_edge_set.edge_dict), 3)

    def test_edge_tracker(self):
        test_tracker = epymorph.draw.EdgeTracker()

        c, d = symbols("c d")

        test_tracker.track_edge("a", "b", c + d)
        test_tracker.track_edge("a", "c", d + d)
        test_tracker.track_edge("a", "b", c * d)

        self.assertEqual(c + d + c * d, test_tracker.edge_dict[("a", "b")])
        self.assertEqual(d + d, test_tracker.edge_dict[("a", "c")])
