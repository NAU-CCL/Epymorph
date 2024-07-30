# pylint: disable=missing-docstring
import unittest

from numpy.testing import assert_array_equal
from sympy import Max

from epymorph import AttributeDef, Shapes, TimeFrame, init, mm_library
from epymorph.compartment_model import CompartmentModel, compartment, edge
from epymorph.database import AbsoluteName
from epymorph.geography.us_census import StateScope
from epymorph.movement.parser import (ALL_DAYS, DailyClause, MovementSpec,
                                      MoveSteps)
from epymorph.movement.parser_util import Duration
from epymorph.rume import (DEFAULT_STRATA, Gpm, MultistrataRume, Rume,
                           SingleStrataRume, combine_tau_steps, remap_taus)
from epymorph.test import EpymorphTestCase


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


# TODO: this test might need to move to compartment_model_test
class CombineIpmTest(unittest.TestCase):
    def test_combine_1(self):
        sir = Sir()

        meta_attributes = [
            AttributeDef("beta_bbb_aaa", float, Shapes.TxN),
        ]

        def meta_edges(s: RumeSymbols):
            [S_aaa, I_aaa, R_aaa] = s.compartments("aaa")
            [S_bbb, I_bbb, R_bbb] = s.compartments("bbb")
            [beta_bbb_aaa] = s.meta_attributes()
            N_aaa = s.total_nonzero("aaa")
            return [
                edge(S_bbb, I_bbb, beta_bbb_aaa * S_bbb * I_aaa / N_aaa),
            ]

        model = combine_ipms(
            strata=[('aaa', sir), ('bbb', sir)],
            meta_attributes=meta_attributes,
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
            model.symbols.compartment_symbols,
            list(symbols("S_aaa I_aaa R_aaa S_bbb I_bbb R_bbb")),
        )

        # Check attribute mapping
        self.assertEqual(
            model.symbols.attribute_symbols,
            list(symbols("beta_aaa gamma_aaa beta_bbb gamma_bbb beta_bbb_aaa_meta")),
        )

        self.assertEqual(
            list(model.attributes.keys()),
            [
                AbsoluteName("gpm:aaa", "ipm", "beta"),
                AbsoluteName("gpm:aaa", "ipm", "gamma"),
                AbsoluteName("gpm:bbb", "ipm", "beta"),
                AbsoluteName("gpm:bbb", "ipm", "gamma"),
                AbsoluteName("meta", "ipm", "beta_bbb_aaa"),
            ],
        )

        self.assertEqual(
            list(model.attributes.values()),
            [
                AttributeDef('beta', float, Shapes.TxN),
                AttributeDef('gamma', float, Shapes.TxN),
                AttributeDef('beta', float, Shapes.TxN),
                AttributeDef('gamma', float, Shapes.TxN),
                AttributeDef('beta_bbb_aaa', float, Shapes.TxN),
            ],
        )

        [S_aaa, I_aaa, R_aaa, S_bbb, I_bbb, R_bbb] = model.symbols.compartment_symbols
        [beta_aaa, gamma_aaa, beta_bbb, gamma_bbb,
            beta_bbb_aaa] = model.symbols.attribute_symbols

        self.assertEqual(model.transitions, [
            edge(S_aaa, I_aaa, rate=beta_aaa * S_aaa * I_aaa),
            edge(I_aaa, R_aaa, rate=gamma_aaa * I_aaa),
            edge(S_bbb, I_bbb, rate=beta_bbb * S_bbb * I_bbb),
            edge(I_bbb, R_bbb, rate=gamma_bbb * I_bbb),
            edge(S_bbb, I_bbb, beta_bbb_aaa * S_bbb *
                 I_aaa / Max(1, S_aaa + I_aaa + R_aaa)),
        ])


class CombineMmTest(EpymorphTestCase):

    def test_combine_tau_steps_1(self):
        new_taus, start_map, stop_map = combine_tau_steps({
            "a": [1 / 3, 2 / 3],
            "b": [1 / 2, 1 / 2],
        })
        self.assertListAlmostEqual(new_taus, [1 / 3, 1 / 6, 1 / 2])
        self.assertDictEqual(start_map, {
            "a": {0: 0, 1: 1},
            "b": {0: 0, 1: 2},
        })
        self.assertDictEqual(stop_map, {
            "a": {0: 0, 1: 2},
            "b": {0: 1, 1: 2},
        })

    def test_combine_tau_steps_2(self):
        new_taus, start_map, stop_map = combine_tau_steps({
            "a": [1 / 3, 2 / 3],
        })
        self.assertListAlmostEqual(new_taus, [1 / 3, 2 / 3])
        self.assertDictEqual(start_map, {
            "a": {0: 0, 1: 1},
        })
        self.assertDictEqual(stop_map, {
            "a": {0: 0, 1: 1},
        })

    def test_combine_tau_steps_3(self):
        new_taus, start_map, stop_map = combine_tau_steps({
            "a": [1 / 3, 2 / 3],
            "b": [1 / 3, 2 / 3],
        })
        self.assertListAlmostEqual(new_taus, [1 / 3, 2 / 3])
        self.assertDictEqual(start_map, {
            "a": {0: 0, 1: 1},
            "b": {0: 0, 1: 1},
        })
        self.assertDictEqual(stop_map, {
            "a": {0: 0, 1: 1},
            "b": {0: 0, 1: 1},
        })

    def test_combine_tau_steps_4(self):
        new_taus, start_map, stop_map = combine_tau_steps({
            "a": [0.5, 0.5],
            "b": [0.2, 0.4, 0.4],
            "c": [0.1, 0.7, 0.2],
            "d": [0.5, 0.5],
        })
        self.assertListAlmostEqual(new_taus, [0.1, 0.1, 0.3, 0.1, 0.2, 0.2])
        self.assertDictEqual(start_map, {
            "a": {0: 0, 1: 3},
            "b": {0: 0, 1: 2, 2: 4},
            "c": {0: 0, 1: 1, 2: 5},
            "d": {0: 0, 1: 3},
        })
        self.assertDictEqual(stop_map, {
            "a": {0: 2, 1: 5},
            "b": {0: 1, 1: 3, 2: 5},
            "c": {0: 0, 1: 4, 2: 5},
            "d": {0: 2, 1: 5},
        })

    def test_remap_taus_1(self):
        mm1 = MovementSpec(
            steps=MoveSteps([1 / 3, 2 / 3]),
            attributes=[],
            predef=None,
            clauses=[
                DailyClause(
                    days=ALL_DAYS,
                    leave_step=0,
                    duration=Duration(0, 'd'),
                    return_step=1,
                    function='place-hodor',
                ),
            ],
        )

        mm2 = MovementSpec(
            steps=MoveSteps([1 / 2, 1 / 2]),
            attributes=[],
            predef=None,
            clauses=[
                DailyClause(
                    days=ALL_DAYS,
                    leave_step=1,
                    duration=Duration(0, 'd'),
                    return_step=1,
                    function='place-hodor',
                ),
            ],
        )

        new_mms = remap_taus([('a', mm1), ('b', mm2)])

        new_taus = new_mms["a"].steps.step_lengths
        self.assertListAlmostEqual(new_taus, [1 / 3, 1 / 6, 1 / 2])
        self.assertEqual(len(new_mms), 2)

        new_mm1 = new_mms['a']
        self.assertEqual(new_mm1.clauses[0].leave_step, 0)
        self.assertEqual(new_mm1.clauses[0].return_step, 2)

        new_mm2 = new_mms['b']
        self.assertEqual(new_mm2.clauses[0].leave_step, 2)
        self.assertEqual(new_mm2.clauses[0].return_step, 2)


class RumeTest(EpymorphTestCase):

    def test_create_monostrata_1(self):
        # A single-strata RUME uses the IPM without modification.
        sir = Sir()
        centroids = mm_library['centroids']()
        # Make sure centroids has the tau steps we will expect later...
        self.assertListAlmostEqual(centroids.steps.step_lengths, [1 / 3, 2 / 3])

        rume = SingleStrataRume.build(
            ipm=sir,
            mm=centroids,
            init=init.NoInfection(),
            scope=StateScope.in_states(['04', '35']),
            time_frame=TimeFrame.of("2021-01-01", 180),
            params={},
        )
        self.assertIs(sir, rume.ipm)

        self.assertEqual(rume.dim.compartments, 3)
        self.assertEqual(rume.dim.events, 2)
        self.assertEqual(rume.dim.days, 180)
        self.assertEqual(rume.dim.ticks, 360)
        self.assertListAlmostEqual(rume.dim.tau_step_lengths, [1 / 3, 2 / 3])
        self.assertEqual(rume.dim.nodes, 2)

        assert_array_equal(
            rume.compartment_mask(DEFAULT_STRATA),
            [True, True, True],
        )
        assert_array_equal(
            rume.compartment_mobility(DEFAULT_STRATA),
            [True, True, True],
        )

    def test_create_multistrata_1(self):
        # Test a multi-strata model.

        sir = Sir()
        no = mm_library['no']()
        # Make sure 'no' has the tau steps we will expect later...
        self.assertListAlmostEqual(no.steps.step_lengths, [1.0])

        rume = MultistrataRume.build(
            strata=[
                Gpm(
                    name="aaa",
                    ipm=sir,
                    mm=no,
                    init=init.SingleLocation(location=0, seed_size=100),
                ),
                Gpm(
                    name="bbb",
                    ipm=sir,
                    mm=no,
                    init=init.SingleLocation(location=0, seed_size=100),
                ),
            ],
            meta_requirements=[],
            meta_edges=lambda _: [],
            scope=StateScope.in_states(['04', '35']),
            time_frame=TimeFrame.of("2021-01-01", 180),
            params={},
        )

        self.assertEqual(rume.dim.compartments, 6)
        self.assertEqual(rume.dim.events, 4)
        self.assertEqual(rume.dim.days, 180)
        self.assertEqual(rume.dim.ticks, 180)
        self.assertListAlmostEqual(rume.dim.tau_step_lengths, [1.0])
        self.assertEqual(rume.dim.nodes, 2)

        assert_array_equal(
            rume.compartment_mask("aaa"),
            [True, True, True, False, False, False],
        )
        assert_array_equal(
            rume.compartment_mask("bbb"),
            [False, False, False, True, True, True],
        )
        assert_array_equal(
            rume.compartment_mobility("aaa"),
            [True, True, True, False, False, False],
        )
        assert_array_equal(
            rume.compartment_mobility("bbb"),
            [False, False, False, True, True, True],
        )

        # NOTE: these tests will break if someone alters the MM or Init definition; even just the comments
        self.assertDictEqual(rume.attributes, {
            AbsoluteName("gpm:aaa", "ipm", "beta"): AttributeDef("beta", float, Shapes.TxN),
            AbsoluteName("gpm:aaa", "ipm", "gamma"): AttributeDef("gamma", float, Shapes.TxN),
            AbsoluteName("gpm:bbb", "ipm", "beta"): AttributeDef("beta", float, Shapes.TxN),
            AbsoluteName("gpm:bbb", "ipm", "gamma"): AttributeDef("gamma", float, Shapes.TxN),
            AbsoluteName("gpm:aaa", "init", "population"): AttributeDef("population", int, Shapes.N,
                                                                        comment="The population at each geo node."),
            AbsoluteName("gpm:bbb", "init", "population"): AttributeDef("population", int, Shapes.N,
                                                                        comment="The population at each geo node."),
        })

    def test_create_multistrata_2(self):
        # Test special case: a multi-strata model but with only one strata.

        sir = Sir()
        centroids = mm_library['centroids']()
        # Make sure centroids has the tau steps we will expect later...
        self.assertListAlmostEqual(centroids.steps.step_lengths, [1 / 3, 2 / 3])

        rume = MultistrataRume.build(
            strata=[
                Gpm(
                    name="aaa",
                    ipm=sir,
                    mm=centroids,
                    init=init.NoInfection(),
                ),
            ],
            meta_requirements=[],
            meta_edges=lambda _: [],
            scope=StateScope.in_states(['04', '35']),
            time_frame=TimeFrame.of("2021-01-01", 180),
            params={},
        )

        self.assertEqual(rume.dim.compartments, 3)
        self.assertEqual(rume.dim.events, 2)
        self.assertEqual(rume.dim.days, 180)
        self.assertEqual(rume.dim.ticks, 360)
        self.assertListAlmostEqual(rume.dim.tau_step_lengths, [1 / 3, 2 / 3])
        self.assertEqual(rume.dim.nodes, 2)

        # NOTE: these tests will break if someone alters the MM or Init definition; even just the comments
        self.assertDictEqual(rume.attributes, {
            AbsoluteName("gpm:aaa", "ipm", "beta"):
                AttributeDef("beta", float, Shapes.TxN),

            AbsoluteName("gpm:aaa", "ipm", "gamma"):
                AttributeDef("gamma", float, Shapes.TxN),

            AbsoluteName("gpm:aaa", "mm", "population"):
                AttributeDef("population", int, Shapes.N,
                             comment="The total population at each node."),

            AbsoluteName("gpm:aaa", "mm", "centroid"):
                AttributeDef("centroid", [('longitude', float), ('latitude', float)], Shapes.N,
                             comment="The centroids for each node as (longitude, latitude) tuples."),

            AbsoluteName("gpm:aaa", "mm", "phi"):
                AttributeDef("phi", float, Shapes.S,
                             comment="Influences the distance that movers tend to travel.",
                             default_value=40.0),

            AbsoluteName("gpm:aaa", "init", "population"):
                AttributeDef("population", int, Shapes.N,
                             comment="The population at each geo node."),
        })
