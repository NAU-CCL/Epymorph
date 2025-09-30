# ruff: noqa: PT009,PT027,N806
import unittest
import warnings
from typing import Sequence

from sympy import Max
from sympy import symbols as sympy_symbols

from epymorph.attribute import AbsoluteName, AttributeDef
from epymorph.compartment_model import (
    BIRTH,
    DEATH,
    CombinedCompartmentModel,
    CompartmentModel,
    CompartmentName,
    EdgeDef,
    EdgeName,
    MultiStrataModelSymbols,
    TransitionDef,
    compartment,
    edge,
)
from epymorph.data_shape import Shapes
from epymorph.error import IPMValidationError
from epymorph.sympy_shim import to_symbol
from epymorph.util import are_instances


class CompartmentModelTest(unittest.TestCase):
    def test_create_01(self):
        class MyIpm(CompartmentModel):
            compartments = [
                compartment("S", tags=["test_tag"]),
                compartment("I"),
                compartment("R"),
            ]

            requirements = [
                AttributeDef("beta", float, Shapes.N),
                AttributeDef("gamma", float, Shapes.N),
            ]

            def edges(self, symbols):
                S, I, R = symbols.compartments("S", "I", "R")
                beta, gamma = symbols.requirements("beta", "gamma")
                return [
                    edge(S, I, rate=beta * S * I),
                    edge(I, R, rate=gamma * I),
                ]

        model = MyIpm()

        self.assertEqual(model.num_compartments, 3)
        self.assertEqual(model.num_events, 2)

        self.assertEqual(
            list(model.compartments),
            [
                compartment("S", ["test_tag"]),
                compartment("I", []),
                compartment("R", []),
            ],
        )
        self.assertEqual(
            list(model.requirements_dict.keys()),
            [
                AbsoluteName("gpm:all", "ipm", "beta"),
                AbsoluteName("gpm:all", "ipm", "gamma"),
            ],
        )
        self.assertEqual(
            list(model.requirements_dict.values()),
            [
                AttributeDef("beta", type=float, shape=Shapes.N),
                AttributeDef("gamma", type=float, shape=Shapes.N),
            ],
        )

        S, I, R = model.symbols.all_compartments
        beta, gamma = model.symbols.all_requirements
        self.assertEqual(
            list(model.transitions),
            [
                edge(S, I, rate=beta * S * I),
                edge(I, R, rate=gamma * I),
            ],
        )

    def test_create_02(self):
        class MyIpm(CompartmentModel):
            compartments = [
                compartment("S"),
                compartment("I"),
                compartment("R"),
            ]
            requirements = [
                AttributeDef("beta", float, Shapes.N),
                AttributeDef("gamma", float, Shapes.N),
                AttributeDef("b", float, Shapes.N),  # birth rate
                AttributeDef("d", float, Shapes.N),  # death rate
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
        # Test for error: Reference an undeclared compartment in a transition.
        with self.assertRaises(IPMValidationError) as e:

            class MyIpm(CompartmentModel):
                compartments = [
                    compartment("S", tags=["test_tag"]),
                    compartment("I"),
                    compartment("R"),
                ]

                requirements = [
                    AttributeDef("beta", float, Shapes.N),
                    AttributeDef("gamma", float, Shapes.N),
                ]

                def edges(self, symbols):
                    S, I, R = symbols.all_compartments
                    beta, gamma = symbols.all_requirements
                    return [
                        edge(S, I, rate=beta * S * I),
                        edge(I, R, rate=gamma * I),
                        edge(I, to_symbol("bad_compartment"), rate=gamma * I),
                    ]

        self.assertIn("missing compartments: bad_compartment", str(e.exception).lower())

    def test_create_04(self):
        # Test for error: Reference an undeclared requirement in a transition.
        with self.assertRaises(IPMValidationError) as e:

            class MyIpm(CompartmentModel):
                compartments = [
                    compartment("S", tags=["test_tag"]),
                    compartment("I"),
                    compartment("R"),
                ]

                requirements = [
                    AttributeDef("beta", float, Shapes.N),
                    AttributeDef("gamma", float, Shapes.N),
                ]

                def edges(self, symbols):
                    S, I, R = symbols.all_compartments
                    beta, gamma = symbols.all_requirements

                    return [
                        edge(S, I, rate=beta * S * I),
                        edge(I, R, rate=gamma * to_symbol("bad_symbol") * I),
                    ]

        self.assertIn("missing requirements: bad_symbol", str(e.exception).lower())

    def test_create_05(self):
        # Test for error: Source and destination are both exogenous!
        with self.assertRaises(IPMValidationError) as e:

            class MyIpm(CompartmentModel):
                compartments = [
                    compartment("S", tags=["test_tag"]),
                    compartment("I"),
                    compartment("R"),
                ]

                requirements = [
                    AttributeDef("beta", float, Shapes.N),
                    AttributeDef("gamma", float, Shapes.N),
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
        with self.assertRaises(IPMValidationError) as e:

            class MyIpm(CompartmentModel):
                compartments = []
                requirements = [
                    AttributeDef("beta", float, Shapes.N),
                    AttributeDef("gamma", float, Shapes.N),
                ]

                def edges(self, symbols):
                    return []

        self.assertIn("invalid compartments", str(e.exception).lower())

    def test_create_07(self):
        # Test for warning: unused requirements.
        with self.assertWarns(Warning) as w:

            class MyIpm(CompartmentModel):
                compartments = [
                    compartment("S", tags=["test_tag"]),
                    compartment("I"),
                    compartment("R"),
                ]
                requirements = [
                    AttributeDef("beta", float, Shapes.N),
                    AttributeDef("gamma", float, Shapes.N),
                    AttributeDef("extraneous", float, Shapes.N),
                ]

                def edges(self, symbols):
                    S, I, R = symbols.all_compartments
                    beta, gamma, extraneous = symbols.all_requirements
                    return [
                        edge(S, I, rate=beta * S * I),
                        edge(I, R, rate=gamma * I),
                    ]

        self.assertIn("extra requirements: extraneous", str(w.warning).lower())

    def test_create_08(self):
        # Test for warning: unused compartments.
        with self.assertWarns(Warning) as w:

            class MyIpm(CompartmentModel):
                compartments = [
                    compartment("S", tags=["test_tag"]),
                    compartment("I"),
                    compartment("R"),
                    # Q is defined, but not used anywhere
                    compartment("Q"),
                ]
                requirements = [
                    AttributeDef("beta", float, Shapes.N),
                    AttributeDef("gamma", float, Shapes.N),
                ]

                def edges(self, symbols):
                    S, I, R, Q = symbols.all_compartments
                    beta, gamma = symbols.all_requirements
                    return [
                        edge(S, I, rate=beta * S * I),
                        edge(I, R, rate=gamma * I),
                    ]

        self.assertIn("extra compartments: q", str(w.warning).lower())

    def test_create_09(self):
        # This is a valid IPM.
        # Don't raise warnings or errors.
        with warnings.catch_warnings(action="error"):

            class TwoSpeciesIPM(CompartmentModel):
                compartments = [
                    compartment("S"),
                    compartment("I"),
                    compartment("R"),
                    # X has no trxs itself, but it is used in a trx expr
                    compartment("X"),
                ]

                requirements = [
                    AttributeDef("beta", float, Shapes.N),
                    AttributeDef("gamma", float, Shapes.N),
                ]

                def edges(self, symbols):
                    S, I, R, X = symbols.all_compartments
                    beta, gamma = symbols.all_requirements
                    return [
                        # infection of S is proportional to the number of X
                        # but there is no edge for X
                        edge(S, I, rate=beta * S * X),
                        edge(I, R, rate=gamma * I),
                    ]

    def test_compartment_name(self):
        # Test for compartment names that include spaces.
        with self.assertRaises(ValueError):
            compartment("some people")

    def test_attribute_name(self):
        # Test for attribute names that include spaces.
        with self.assertRaises(ValueError):
            AttributeDef("some attribute", float, Shapes.N)

    def test_combined_01(self):
        class Sir(CompartmentModel):
            compartments = [
                compartment("S"),
                compartment("I"),
                compartment("R"),
            ]

            requirements = [
                AttributeDef("beta", float, Shapes.TxN),
                AttributeDef("gamma", float, Shapes.TxN),
            ]

            def edges(self, symbols):
                S, I, R = symbols.all_compartments
                beta, gamma = symbols.all_requirements
                return [
                    edge(S, I, rate=beta * S * I),
                    edge(I, R, rate=gamma * I),
                ]

        sir = Sir()

        def meta_edges(sym: MultiStrataModelSymbols):
            [S_aaa, I_aaa, R_aaa] = sym.strata_compartments("aaa")
            [S_bbb, I_bbb, R_bbb] = sym.strata_compartments("bbb")
            [beta_bbb_aaa] = sym.all_meta_requirements
            N_aaa = Max(1, S_aaa + I_aaa + R_aaa)
            return [
                edge(S_bbb, I_bbb, beta_bbb_aaa * S_bbb * I_aaa / N_aaa),
            ]

        model = CombinedCompartmentModel(
            strata=[("aaa", sir), ("bbb", sir)],
            meta_requirements=[
                AttributeDef("beta_bbb_aaa", float, Shapes.TxN),
            ],
            meta_edges=meta_edges,
        )

        self.assertEqual(model.num_compartments, 6)
        self.assertEqual(model.num_events, 5)

        # Check compartment mapping
        self.assertEqual(
            [c.name.full for c in model.compartments],
            ["S_aaa", "I_aaa", "R_aaa", "S_bbb", "I_bbb", "R_bbb"],
        )

        self.assertEqual(
            model.symbols.all_compartments,
            list(sympy_symbols("S_aaa I_aaa R_aaa S_bbb I_bbb R_bbb")),
        )

        self.assertEqual(
            model.symbols.strata_compartments("aaa"),
            list(sympy_symbols("S_aaa I_aaa R_aaa")),
        )

        self.assertEqual(
            model.symbols.strata_compartments("bbb"),
            list(sympy_symbols("S_bbb I_bbb R_bbb")),
        )

        # Check requirement mapping
        self.assertEqual(
            model.symbols.all_requirements,
            list(
                sympy_symbols("beta_aaa gamma_aaa beta_bbb gamma_bbb beta_bbb_aaa_meta")
            ),
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
                AttributeDef("beta", float, Shapes.TxN),
                AttributeDef("gamma", float, Shapes.TxN),
                AttributeDef("beta", float, Shapes.TxN),
                AttributeDef("gamma", float, Shapes.TxN),
                AttributeDef("beta_bbb_aaa", float, Shapes.TxN),
            ],
        )

        [S_aaa, I_aaa, R_aaa, S_bbb, I_bbb, R_bbb] = model.symbols.all_compartments
        [beta_aaa, gamma_aaa, beta_bbb, gamma_bbb, beta_bbb_aaa] = (
            model.symbols.all_requirements
        )

        s_to_i = EdgeName(
            CompartmentName("S", None, None),
            CompartmentName("I", None, None),
        )
        i_to_r = EdgeName(
            CompartmentName("I", None, None),
            CompartmentName("R", None, None),
        )

        self.assertEqual(
            model.transitions,
            [
                EdgeDef(
                    s_to_i.with_strata("aaa"),
                    beta_aaa * S_aaa * I_aaa,
                    S_aaa,
                    I_aaa,
                ),
                EdgeDef(
                    i_to_r.with_strata("aaa"),
                    gamma_aaa * I_aaa,
                    I_aaa,
                    R_aaa,
                ),
                EdgeDef(
                    s_to_i.with_strata("bbb"),
                    beta_bbb * S_bbb * I_bbb,
                    S_bbb,
                    I_bbb,
                ),
                EdgeDef(
                    i_to_r.with_strata("bbb"),
                    gamma_bbb * I_bbb,
                    I_bbb,
                    R_bbb,
                ),
                EdgeDef(
                    s_to_i.with_strata("bbb"),
                    beta_bbb_aaa * S_bbb * I_aaa / Max(1, S_aaa + I_aaa + R_aaa),
                    S_bbb,
                    I_bbb,
                ),
            ],
        )

    def test_combined_02(self):
        # Test a combined model using exogenous classes.
        class SirsBirthDeath(CompartmentModel):
            compartments = [
                compartment("S"),
                compartment("I"),
                compartment("R"),
            ]

            requirements = [
                AttributeDef("beta", type=float, shape=Shapes.TxN),
                AttributeDef("gamma", type=float, shape=Shapes.TxN),
                AttributeDef("xi", type=float, shape=Shapes.TxN),
                AttributeDef("birth_rate", type=float, shape=Shapes.TxN),
                AttributeDef("death_rate", type=float, shape=Shapes.TxN),
            ]

            def edges(self, symbols):
                [S, I, R] = symbols.all_compartments
                [b, g, x, br, dr] = symbols.all_requirements
                N = Max(1, S + I + R)
                return [
                    edge(S, I, rate=b * S * I / N),
                    edge(I, R, rate=g * I),
                    edge(R, S, rate=x * R),
                    edge(BIRTH, S, rate=br * N),
                    edge(S, DEATH, rate=dr * S),
                    edge(I, DEATH, rate=dr * I),
                    edge(R, DEATH, rate=dr * R),
                ]

        strata = [("one", SirsBirthDeath()), ("two", SirsBirthDeath())]
        meta_requirements = ()

        def meta_edges(symbols: MultiStrataModelSymbols) -> Sequence[TransitionDef]:
            [S1, I1, R1] = symbols.strata_compartments("one")
            [S2, I2, R2] = symbols.strata_compartments("two")
            [b1, *_] = symbols.strata_requirements("one")
            [b2, *_] = symbols.strata_requirements("two")
            N1 = Max(1, S1 + I1 + R1)
            N2 = Max(1, S2 + I2 + R2)
            return [
                edge(S1, I1, rate=b1 * S1 * I2 / N2),
                edge(S2, I2, rate=b2 * S2 * I1 / N1),
            ]

        multi_ipm = CombinedCompartmentModel(strata, meta_requirements, meta_edges)

        self.assertEqual(
            [x.name.full for x in multi_ipm.compartments],
            ["S_one", "I_one", "R_one", "S_two", "I_two", "R_two"],
        )

        n = CompartmentName
        self.assertTrue(are_instances([*multi_ipm.transitions], EdgeDef))
        self.assertEqual(
            [
                (x.name.compartment_from, x.name.compartment_to)
                for x in multi_ipm.transitions
                if isinstance(x, EdgeDef)
            ],
            [
                # base model for one
                (n("S", None, "one"), n("I", None, "one")),
                (n("I", None, "one"), n("R", None, "one")),
                (n("R", None, "one"), n("S", None, "one")),
                (n("birth", "exogenous", None), n("S", None, "one")),
                (n("S", None, "one"), n("death", "exogenous", None)),
                (n("I", None, "one"), n("death", "exogenous", None)),
                (n("R", None, "one"), n("death", "exogenous", None)),
                # base model for two
                (n("S", None, "two"), n("I", None, "two")),
                (n("I", None, "two"), n("R", None, "two")),
                (n("R", None, "two"), n("S", None, "two")),
                (n("birth", "exogenous", None), n("S", None, "two")),
                (n("S", None, "two"), n("death", "exogenous", None)),
                (n("I", None, "two"), n("death", "exogenous", None)),
                (n("R", None, "two"), n("death", "exogenous", None)),
                # meta edges
                (n("S", None, "one"), n("I", None, "one")),
                (n("S", None, "two"), n("I", None, "two")),
            ],
        )


class _TestSir(CompartmentModel):
    compartments = [
        compartment("S", tags=["test_tag"]),
        compartment("I"),
        compartment("R"),
    ]

    requirements = [
        AttributeDef("beta", float, Shapes.N),
        AttributeDef("gamma", float, Shapes.N),
    ]

    def edges(self, symbols):
        S, I, R = symbols.compartments("S", "I", "R")
        beta, gamma = symbols.requirements("beta", "gamma")
        return [
            edge(S, I, rate=beta * S * I),
            edge(I, R, rate=gamma * I),
        ]


class QuantityStrategyTest(unittest.TestCase):
    def test_select_compartments(self):
        ipm = _TestSir()
        self.assertEqual(
            ipm.select.compartments().labels,
            ["S", "I", "R"],
        )
        self.assertEqual(
            ipm.select.compartments("S").labels,
            ["S"],
        )
        self.assertEqual(
            ipm.select.compartments("S*").labels,
            ["S"],
        )
        self.assertEqual(
            ipm.select.compartments("S", "R").labels,
            ["S", "R"],
        )
        self.assertEqual(
            ipm.select.compartments("*").labels,
            ["S", "I", "R"],
        )

    def test_select_nonmatching_compartment(self):
        ipm = _TestSir()
        with self.assertRaises(ValueError):
            ipm.select.compartments("S", "X")

    def test_select_events(self):
        ipm = _TestSir()
        self.assertEqual(
            ipm.select.events().labels,
            ["S → I", "I → R"],
        )
        self.assertEqual(
            ipm.select.events("S->I").labels,
            ["S → I"],
        )
        self.assertEqual(
            ipm.select.events("*->I").labels,
            ["S → I"],
        )
        self.assertEqual(
            ipm.select.events("S->*").labels,
            ["S → I"],
        )
        self.assertEqual(
            ipm.select.events("*->*").labels,
            ["S → I", "I → R"],
        )
        self.assertEqual(
            ipm.select.events("S->I*").labels,
            ["S → I"],
        )
        self.assertEqual(
            ipm.select.events("S*->*").labels,
            ["S → I"],
        )

    def test_select_nonmatching_event(self):
        ipm = _TestSir()
        with self.assertRaises(ValueError):
            ipm.select.events("S->X")

    def test_select_by(self):
        ipm = _TestSir()
        self.assertEqual(
            ipm.select.by(compartments="S", events="S->I").labels,
            ["S", "S → I"],
        )
        self.assertEqual(
            ipm.select.by(compartments="S").labels,
            ["S"],
        )
        self.assertEqual(
            ipm.select.by(events="S->I").labels,
            ["S → I"],
        )
        self.assertEqual(
            ipm.select.by(compartments=["S", "I"], events=["S->I", "I->R"]).labels,
            ["S", "I", "S → I", "I → R"],
        )


class QuantitySelectionTest(unittest.TestCase):
    def test_compartment_index(self):
        ipm = _TestSir()
        self.assertEqual(1, ipm.select.compartments("I").compartment_index)
        with self.assertRaises(ValueError):
            ipm.select.compartments("I", "R").compartment_index
        with self.assertRaises(ValueError):
            ipm.select.compartments().compartment_index

    def test_compartment_indices(self):
        ipm = _TestSir()
        self.assertEqual((0, 2), ipm.select.compartments("S", "R").compartment_indices)

    def test_event_index(self):
        ipm = _TestSir()
        self.assertEqual(0, ipm.select.events("S->I").event_index)
        with self.assertRaises(ValueError):
            ipm.select.events("S->I", "I->R").event_index
        with self.assertRaises(ValueError):
            ipm.select.events().event_index

    def test_event_indices(self):
        ipm = _TestSir()
        self.assertEqual((0, 1), ipm.select.events("S->I", "I->R").event_indices)
