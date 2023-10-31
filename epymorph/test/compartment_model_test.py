# type: ignore
# pylint: disable=missing-docstring
import unittest

from epymorph.attribute import AttributeDef
from epymorph.compartment_model import (BIRTH, DEATH, CompartmentDef,
                                        IpmAttributeDef, compartment,
                                        create_model, create_symbols, edge,
                                        param)
from epymorph.data_shape import Shapes


# TODO: (refactor) this isn't a very interesting test anymore...
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
            IpmAttributeDef(AttributeDef('beta', Shapes.S,
                            float, 'params'), beta, True),
            IpmAttributeDef(AttributeDef('gamma', Shapes.S,
                            float, 'params'), gamma, True),
        ])

        # self.assertTrue(np.array_equal(
        #     model.apply_matrix,
        #     np.array([[-1, +1, 0], [0, -1, +1]], dtype=SimDType)
        # ))

        # self.assertEqual(model.events_leaving_compartment, [[0], [1], []])
        # self.assertEqual(model.source_compartment_for_event, [0, 1])

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

        # self.assertTrue(np.array_equal(
        #     model.apply_matrix,
        #     np.array([
        #         [-1, +1, 0],
        #         [+1, 0, 0],
        #         [0, -1, +1],
        #         [-1, 0, 0],
        #         [0, -1, 0],
        #         [0, 0, -1],
        #     ], dtype=SimDType)
        # ))

        # self.assertEqual(model.source_compartment_for_event, [0, -1, 1, 0, 1, 2])
