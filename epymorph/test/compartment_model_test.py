# pylint: disable=missing-docstring,unused-variable
import unittest

from epymorph.compartment_model import (BIRTH, DEATH, CompartmentDef,
                                        CompartmentModel, compartment, edge)
from epymorph.data_shape import Shapes
from epymorph.database import AbsoluteName
from epymorph.error import IpmValidationException
from epymorph.simulation import AttributeDef
from epymorph.sympy_shim import to_symbol


class CompartmentModelTest(unittest.TestCase):

    def test_create_01(self):

        class MyIpm(CompartmentModel):
            compartments = [
                compartment('S', tags=['test_tag']),
                compartment('I'),
                compartment('R'),
            ]

            requirements = [
                AttributeDef('beta', float, Shapes.N),
                AttributeDef('gamma', float, Shapes.N),
            ]

            def edges(self, symbols):
                S, I, R = symbols.compartments('S', 'I', 'R')
                beta, gamma = symbols.requirements('beta', 'gamma')
                return [
                    edge(S, I, rate=beta * S * I),
                    edge(I, R, rate=gamma * I),
                ]

        model = MyIpm()

        self.assertEqual(model.num_compartments, 3)
        self.assertEqual(model.num_events, 2)

        self.assertEqual(model.compartments, [
            CompartmentDef('S', ['test_tag']),
            CompartmentDef('I', []),
            CompartmentDef('R', []),
        ])
        self.assertEqual(list(model.requirements_dict.keys()), [
            AbsoluteName("gpm:all", "ipm", "beta"),
            AbsoluteName("gpm:all", "ipm", "gamma"),
        ])
        self.assertEqual(list(model.requirements_dict.values()), [
            AttributeDef('beta', type=float, shape=Shapes.N),
            AttributeDef('gamma', type=float, shape=Shapes.N),
        ])

        S, I, R = model.symbols.all_compartments
        beta, gamma = model.symbols.all_requirements
        self.assertEqual(model.transitions, [
            edge(S, I, rate=beta * S * I),
            edge(I, R, rate=gamma * I),
        ])

    def test_create_02(self):

        class MyIpm(CompartmentModel):
            compartments = [
                compartment('S'),
                compartment('I'),
                compartment('R'),
            ]
            requirements = [
                AttributeDef('beta', float, Shapes.N),
                AttributeDef('gamma', float, Shapes.N),
                AttributeDef('b', float, Shapes.N),  # birth rate
                AttributeDef('d', float, Shapes.N),  # death rate
            ]

            def edges(self, symbols):
                S, I, R = symbols.all_compartments
                beta, gamma, b, d = symbols.all_requirements
                return [
                    edge(S, I, rate=beta * S * I),
                    edge(BIRTH, S, rate=b),
                    edge(I, R, rate=gamma * I),
                    edge(S, DEATH, rate=d * S),
                    edge(I, DEATH, rate=d * I),
                    edge(R, DEATH, rate=d * R),
                ]

        model = MyIpm()

        self.assertEqual(model.num_compartments, 3)
        self.assertEqual(model.num_events, 6)

    def test_create_03(self):
        # Test for error: Attempt to reference an undeclared compartment in a transition.
        with self.assertRaises(IpmValidationException):
            class MyIpm(CompartmentModel):
                compartments = [
                    compartment('S', tags=['test_tag']),
                    compartment('I'),
                    compartment('R'),
                ]

                requirements = [
                    AttributeDef('beta', float, Shapes.N),
                    AttributeDef('gamma', float, Shapes.N),
                ]

                def edges(self, symbols):
                    S, I, R = symbols.all_compartments
                    beta, gamma = symbols.all_requirements
                    return [
                        edge(S, I, rate=beta * S * I),
                        edge(I, R, rate=gamma * I),
                        edge(I, to_symbol('bad_compartment'), rate=gamma * I),
                    ]

    def test_create_04(self):
        # Test for error: Attempt to reference an undeclared attribute in a transition.
        with self.assertRaises(IpmValidationException):
            class MyIpm(CompartmentModel):
                compartments = [
                    compartment('S', tags=['test_tag']),
                    compartment('I'),
                    compartment('R'),
                ]

                requirements = [
                    AttributeDef('beta', float, Shapes.N),
                    AttributeDef('gamma', float, Shapes.N),
                ]

                def edges(self, symbols):
                    S, I, R = symbols.all_compartments
                    beta, gamma = symbols.all_requirements

                    return [
                        edge(S, I, rate=beta * S * I),
                        edge(I, R, rate=gamma * to_symbol('bad_symbol') * I),
                    ]

    def test_create_05(self):
        # Test for error: Source and destination are both exogenous!
        with self.assertRaises(IpmValidationException):
            class MyIpm(CompartmentModel):
                compartments = [
                    compartment('S', tags=['test_tag']),
                    compartment('I'),
                    compartment('R'),
                ]

                requirements = [
                    AttributeDef('beta', float, Shapes.N),
                    AttributeDef('gamma', float, Shapes.N),
                ]

                def edges(self, symbols):
                    S, I, R = symbols.all_compartments
                    beta, gamma = symbols.all_requirements
                    return [
                        edge(S, I, rate=beta * S * I),
                        edge(I, R, rate=gamma * I),
                        edge(BIRTH, DEATH, rate=100),
                    ]

    def test_create_06(self):
        # Test for error: model with no compartments.
        with self.assertRaises(IpmValidationException):
            class MyIpm(CompartmentModel):
                compartments = []
                requirements = [
                    AttributeDef('beta', float, Shapes.N),
                    AttributeDef('gamma', float, Shapes.N),
                ]

                def edges(self, symbols):
                    return []
