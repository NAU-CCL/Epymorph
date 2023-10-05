import unittest

import numpy as np

from epymorph.context import SimDType
from epymorph.data_shape import Shapes
from epymorph.ipm.attribute import ParamDef, param
from epymorph.ipm.compartment_model import (BIRTH, DEATH, CompartmentDef,
                                            compartment, create_model,
                                            create_symbols, edge)


class CompartmentModelTest(unittest.TestCase):

    def test_create_01(self):
        symbols = create_symbols(
            compartments=[
                compartment('S'),
                compartment('I'),
                compartment('R'),
            ],
            attributes=[
                param('beta'),
                param('gamma'),
            ]
        )

        [S, I, R] = symbols.compartment_symbols
        [beta, gamma] = symbols.attribute_symbols

        model = create_model(
            symbols=symbols,
            transitions=[
                edge(S, I, rate=beta * S * I),
                edge(I, R, rate=gamma * I),
            ],
        )

        self.assertEqual(model.num_compartments, 3)
        self.assertEqual(model.num_events, 2)

        self.assertEqual(model.transitions, [
            edge(S, I, rate=beta * S * I),
            edge(I, R, rate=gamma * I),
        ])
        self.assertEqual(model.compartments, [
            CompartmentDef(S, 'S', []),
            CompartmentDef(I, 'I', []),
            CompartmentDef(R, 'R', []),
        ])
        self.assertEqual(model.attributes, [
            ParamDef(beta, 'beta', Shapes.S, float, True),
            ParamDef(gamma, 'gamma', Shapes.S, float, True),
        ])

        self.assertTrue(np.array_equal(
            model.apply_matrix,
            np.array([[-1, +1, 0], [0, -1, +1]], dtype=SimDType)
        ))

        self.assertEqual(model.events_leaving_compartment, [[0], [1], []])
        self.assertEqual(model.source_compartment_for_event, [0, 1])

    def test_create_02(self):
        symbols = create_symbols(
            compartments=[
                compartment('S'),
                compartment('I'),
                compartment('R'),
            ],
            attributes=[
                param('beta'),
                param('gamma'),
                param('b'),  # birth rate
                param('d'),  # death rate
            ]
        )

        [S, I, R] = symbols.compartment_symbols
        [beta, gamma, b, d] = symbols.attribute_symbols

        model = create_model(
            symbols=symbols,
            transitions=[
                edge(S, I, rate=beta * S * I),
                edge(BIRTH, S, rate=b),
                edge(I, R, rate=gamma * I),
                edge(S, DEATH, rate=d * S),
                edge(I, DEATH, rate=d * I),
                edge(R, DEATH, rate=d * R),
            ],
        )

        self.assertEqual(model.num_compartments, 3)
        self.assertEqual(model.num_events, 6)

        self.assertTrue(np.array_equal(
            model.apply_matrix,
            np.array([
                [-1, +1, 0],
                [+1, 0, 0],
                [0, -1, +1],
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, -1],
            ], dtype=SimDType)
        ))

        self.assertEqual(model.source_compartment_for_event, [0, -1, 1, 0, 1, 2])
