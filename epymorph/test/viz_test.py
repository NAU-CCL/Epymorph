import unittest
import epymorph.viz
from epymorph.compartment_model import CompartmentModel
from epymorph import ipm_library
from graphviz import Digraph
from sympy import Symbol
from os import path, remove, rmdir

class VizTest(unittest.TestCase):
    test_ipm = ipm_library['sirs']()
    test_graph = epymorph.viz.build_ipm_graph(test_ipm)

    def test_build_model(self):
        edge_prefixes = ['\tS -> I', '\tI -> R', '\tR -> S']

        graph_attrs = self.test_graph.__dict__

        self.assertEqual(graph_attrs['_format'], 'png')
        self.assertEqual(graph_attrs['_encoding'], 'utf-8')
        self.assertEqual(graph_attrs['filename'], 'Digraph.gv')
        self.assertEqual(graph_attrs['graph_attr']['rankdir'], 'LR')
        self.assertEqual(graph_attrs['node_attr']['shape'], 'square')
        self.assertEqual(graph_attrs['node_attr']['width'], '.9')
        self.assertEqual(graph_attrs['node_attr']['height'], '.8')
        self.assertEqual(graph_attrs['edge_attr']['minlen'], '2.0')

        for edge_index in range(len(graph_attrs['body'])):
            self.assertTrue(graph_attrs['body'][edge_index].startswith(edge_prefixes[edge_index]))

    def test_edge_tracker(self):
        test_tracker = epymorph.viz.EdgeTracker()

        c = Symbol('c')
        d = Symbol('d')

        test_tracker.track_edge('a', 'b', c + d)
        test_tracker.track_edge('a', 'c', d + d)
        test_tracker.track_edge('a', 'b', c * d)

        self.assertEqual(c + d + c * d, test_tracker.get_edge_label('a', 'b'))
        self.assertEqual(d + d, test_tracker.get_edge_label('a', 'c'))

    def test_pngtolabel(self):
        self.assertEqual(epymorph.viz.png_to_label('test_path'), 
        '<<TABLE border="0"><TR><TD><IMG SRC="test_path"/></TD></TR></TABLE>>')

    def test_save(self):
        self.assertFalse(epymorph.viz.save_model(self.test_graph, ''))

        self.assertTrue(epymorph.viz.save_model(self.test_graph, 'test_model'))

        self.assertTrue(path.exists('model_pngs/test_model.png'))

        remove('model_pngs/test_model.png')

        rmdir('model_pngs')

