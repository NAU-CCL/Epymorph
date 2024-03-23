import unittest
from os import path, remove, rmdir

from graphviz import Digraph
from sympy import Max, symbols

import epymorph.draw
from epymorph import ipm_library
from epymorph.compartment_model import CompartmentModel


class DrawTest(unittest.TestCase):

    def test_build_edges(self):
        test_ipm = ipm_library['sirs']()
        test_edge_set = epymorph.draw.build_ipm_edge_set(test_ipm)

        S, I, R, beta, gamma, xi = symbols('S I R beta gamma xi')

        N = Max(1, S + I + R)

        self.assertEqual(test_edge_set.edge_dict[('S', 'I')], beta * S * I / N)
        self.assertEqual(test_edge_set.edge_dict[('I', 'R')], gamma * I)
        self.assertEqual(test_edge_set.edge_dict[('R', 'S')], xi * R)

    def test_edge_tracker(self):
        test_tracker = epymorph.draw.EdgeTracker()

        c, d = symbols('c d')

        test_tracker.track_edge('a', 'b', c + d)
        test_tracker.track_edge('a', 'c', d + d)
        test_tracker.track_edge('a', 'b', c * d)

        self.assertEqual(c + d + c * d, test_tracker.edge_dict[('a', 'b')])
        self.assertEqual(d + d, test_tracker.edge_dict[('a', 'c')])

    def test_save(self):
        # if model_pngs didnt exist before the test, delete it
        remove_dir_check = not path.exists('model_pngs')

        test_graph = Digraph(format='png')

        test_graph.edge("a", "b", "label")

        self.assertFalse(epymorph.draw.save_model(test_graph, ''))

        self.assertTrue(epymorph.draw.save_model(test_graph, 'test_model'))

        self.assertTrue(path.exists('model_pngs/test_model.png'))

        remove('model_pngs/test_model.png')

        # delete directory if it didnt exist before test
        if remove_dir_check:
            rmdir('model_pngs')
