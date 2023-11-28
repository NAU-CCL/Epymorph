# type: ignore
# pylint: disable=missing-docstring
import unittest

from epymorph.compartment_model import (BIRTH, DEATH, CompartmentDef,
                                        IpmAttributeDef, compartment,
                                        create_model, create_symbols, edge,
                                        param)
from epymorph.data_shape import Shapes
from epymorph.error import IpmValidationException
from epymorph.sympy_shim import to_symbol


class CompartmentModelTest(unittest.TestCase):

    def test_create_01(self):
        symbols = create_symbols(
            compartments=[
                compartment('S', tags=['test_tag']),
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
            CompartmentDef(S, 'S', ['test_tag']),
            CompartmentDef(I, 'I', []),
            CompartmentDef(R, 'R', []),
        ])
        self.assertEqual(model.attributes, [
            IpmAttributeDef('beta', Shapes.S, float, 'params', beta),
            IpmAttributeDef('gamma', Shapes.S, float, 'params', gamma),
        ])

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

    def test_create_03(self):
        # Test for error: Attempt to reference an undeclared compartment in a transition.
        symbols = create_symbols(
            compartments=[
                compartment('S', tags=['test_tag']),
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

        with self.assertRaises(IpmValidationException):
            create_model(
                symbols=symbols,
                transitions=[
                    edge(S, I, rate=beta * S * I),
                    edge(I, R, rate=gamma * I),
                    edge(I, to_symbol('bad_compartment'), rate=gamma * I),
                ],
            )

    def test_create_04(self):
        # Test for error: Attempt to reference an undeclared attribute in a transition.
        symbols = create_symbols(
            compartments=[
                compartment('S', tags=['test_tag']),
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

        with self.assertRaises(IpmValidationException):
            create_model(
                symbols=symbols,
                transitions=[
                    edge(S, I, rate=beta * S * I),
                    edge(I, R, rate=gamma * to_symbol('bad_symbol') * I),
                ],
            )

    def test_create_05(self):
        # Test for error: Source and destination are both exogenous!
        symbols = create_symbols(
            compartments=[
                compartment('S', tags=['test_tag']),
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

        with self.assertRaises(IpmValidationException):
            create_model(
                symbols=symbols,
                transitions=[
                    edge(S, I, rate=beta * S * I),
                    edge(I, R, rate=gamma * I),
                    edge(BIRTH, DEATH, rate=100),
                ],
            )

    def test_create_06(self):
        # Test for error: model with no compartments.
        symbols = create_symbols(
            compartments=[],
            attributes=[
                param('beta'),
                param('gamma'),
            ]
        )

        with self.assertRaises(IpmValidationException):
            create_model(
                symbols=symbols,
                transitions=[],
            )
