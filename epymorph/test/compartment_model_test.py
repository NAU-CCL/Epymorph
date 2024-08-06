# pylint: disable=missing-docstring,unused-variable
import unittest

from sympy import Max
from sympy import symbols as sympy_symbols

from epymorph.compartment_model import (BIRTH, DEATH, CombinedCompartmentModel,
                                        CompartmentDef, CompartmentModel,
                                        MultistrataModelSymbols, compartment,
                                        edge)
from epymorph.data_shape import Shapes
from epymorph.database import AbsoluteName
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

        self.assertEqual(list(model.compartments), [
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
        self.assertEqual(list(model.transitions), [
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
        with self.assertRaises(TypeError) as e:
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
        self.assertIn("missing compartments: bad_compartment", str(e.exception).lower())

    def test_create_04(self):
        # Test for error: Attempt to reference an undeclared requirement in a transition.
        with self.assertRaises(TypeError) as e:
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
        self.assertIn("missing requirements: bad_symbol", str(e.exception).lower())

    def test_create_05(self):
        # Test for error: Source and destination are both exogenous!
        with self.assertRaises(TypeError) as e:
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
        self.assertIn("both source and destination", str(e.exception).lower())

    def test_create_06(self):
        # Test for error: model with no compartments.
        with self.assertRaises(TypeError) as e:
            class MyIpm(CompartmentModel):
                compartments = []
                requirements = [
                    AttributeDef('beta', float, Shapes.N),
                    AttributeDef('gamma', float, Shapes.N),
                ]

                def edges(self, symbols):
                    return []
        self.assertIn("invalid compartments", str(e.exception).lower())

    def test_combined_01(self):
        class Sir(CompartmentModel):
            compartments = [
                compartment('S'),
                compartment('I'),
                compartment('R'),
            ]

            requirements = [
                AttributeDef('beta', float, Shapes.TxN),
                AttributeDef('gamma', float, Shapes.TxN),
            ]

            def edges(self, symbols):
                S, I, R = symbols.all_compartments
                beta, gamma = symbols.all_requirements
                return [
                    edge(S, I, rate=beta * S * I),
                    edge(I, R, rate=gamma * I),
                ]

        sir = Sir()

        def meta_edges(sym: MultistrataModelSymbols):
            [S_aaa, I_aaa, R_aaa] = sym.strata_compartments("aaa")
            [S_bbb, I_bbb, R_bbb] = sym.strata_compartments("bbb")
            [beta_bbb_aaa] = sym.all_meta_requirements
            N_aaa = Max(1, S_aaa + I_aaa + R_aaa)
            return [
                edge(S_bbb, I_bbb, beta_bbb_aaa * S_bbb * I_aaa / N_aaa),
            ]

        model = CombinedCompartmentModel(
            strata=[('aaa', sir), ('bbb', sir)],
            meta_requirements=[
                AttributeDef("beta_bbb_aaa", float, Shapes.TxN),
            ],
            meta_edges=meta_edges,
        )

        self.assertEqual(model.num_compartments, 6)
        self.assertEqual(model.num_events, 5)

        # Check compartment mapping
        self.assertEqual(
            [c.name for c in model.compartments],
            ['S_aaa', 'I_aaa', 'R_aaa', 'S_bbb', 'I_bbb', 'R_bbb'],
        )

        self.assertEqual(
            model.symbols.all_compartments,
            list(sympy_symbols("S_aaa I_aaa R_aaa S_bbb I_bbb R_bbb")),
        )

        self.assertEqual(
            model.symbols.strata_compartments("aaa"),
            list(sympy_symbols("S_aaa I_aaa R_aaa"))
        )

        self.assertEqual(
            model.symbols.strata_compartments("bbb"),
            list(sympy_symbols("S_bbb I_bbb R_bbb"))
        )

        # Check requirement mapping
        self.assertEqual(
            model.symbols.all_requirements,
            list(sympy_symbols("beta_aaa gamma_aaa beta_bbb gamma_bbb beta_bbb_aaa_meta")),
        )

        self.assertEqual(
            model.symbols.strata_requirements("aaa"),
            list(sympy_symbols("beta_aaa gamma_aaa")),
        )

        self.assertEqual(
            model.symbols.strata_requirements("bbb"),
            list(sympy_symbols("beta_bbb gamma_bbb")),
        )

        self.assertEqual(
            model.symbols.all_meta_requirements,
            [sympy_symbols("beta_bbb_aaa_meta")],
        )

        self.assertEqual(
            list(model.requirements_dict.keys()),
            [
                AbsoluteName("gpm:aaa", "ipm", "beta"),
                AbsoluteName("gpm:aaa", "ipm", "gamma"),
                AbsoluteName("gpm:bbb", "ipm", "beta"),
                AbsoluteName("gpm:bbb", "ipm", "gamma"),
                AbsoluteName("meta", "ipm", "beta_bbb_aaa"),
            ],
        )

        self.assertEqual(
            list(model.requirements_dict.values()),
            [
                AttributeDef('beta', float, Shapes.TxN),
                AttributeDef('gamma', float, Shapes.TxN),
                AttributeDef('beta', float, Shapes.TxN),
                AttributeDef('gamma', float, Shapes.TxN),
                AttributeDef('beta_bbb_aaa', float, Shapes.TxN),
            ],
        )

        [S_aaa, I_aaa, R_aaa, S_bbb, I_bbb, R_bbb] = model.symbols.all_compartments
        [beta_aaa, gamma_aaa, beta_bbb, gamma_bbb,
            beta_bbb_aaa] = model.symbols.all_requirements

        self.assertEqual(model.transitions, [
            edge(S_aaa, I_aaa, rate=beta_aaa * S_aaa * I_aaa),
            edge(I_aaa, R_aaa, rate=gamma_aaa * I_aaa),
            edge(S_bbb, I_bbb, rate=beta_bbb * S_bbb * I_bbb),
            edge(I_bbb, R_bbb, rate=gamma_bbb * I_bbb),
            edge(S_bbb, I_bbb, beta_bbb_aaa * S_bbb *
                 I_aaa / Max(1, S_aaa + I_aaa + R_aaa)),
        ])
