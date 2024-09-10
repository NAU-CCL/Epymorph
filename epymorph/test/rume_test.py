# pylint: disable=missing-docstring
import numpy as np
from numpy.testing import assert_array_equal
from numpy.typing import NDArray

from epymorph import AttributeDef, Shapes, init, mm_library
from epymorph.compartment_model import CompartmentModel, compartment, edge
from epymorph.database import AbsoluteName
from epymorph.geography.us_census import StateScope
from epymorph.movement_model import EveryDay, MovementClause, MovementModel
from epymorph.rume import (
    DEFAULT_STRATA,
    Gpm,
    MultistrataRume,
    SingleStrataRume,
    combine_tau_steps,
    remap_taus,
)
from epymorph.simulation import Tick, TickDelta, TickIndex
from epymorph.test import EpymorphTestCase
from epymorph.time import TimeFrame


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
        [S, I, R] = symbols.all_compartments
        [beta, gamma] = symbols.all_requirements
        return [
            edge(S, I, rate=beta * S * I),
            edge(I, R, rate=gamma * I),
        ]


class CombineMmTest(EpymorphTestCase):
    def test_combine_tau_steps_1(self):
        new_taus, start_map, stop_map = combine_tau_steps(
            {
                "a": [1 / 3, 2 / 3],
                "b": [1 / 2, 1 / 2],
            }
        )
        self.assertListAlmostEqual(new_taus, [1 / 3, 1 / 6, 1 / 2])
        self.assertDictEqual(
            start_map,
            {
                "a": {0: 0, 1: 1},
                "b": {0: 0, 1: 2},
            },
        )
        self.assertDictEqual(
            stop_map,
            {
                "a": {0: 0, 1: 2},
                "b": {0: 1, 1: 2},
            },
        )

    def test_combine_tau_steps_2(self):
        new_taus, start_map, stop_map = combine_tau_steps(
            {
                "a": [1 / 3, 2 / 3],
            }
        )
        self.assertListAlmostEqual(new_taus, [1 / 3, 2 / 3])
        self.assertDictEqual(
            start_map,
            {
                "a": {0: 0, 1: 1},
            },
        )
        self.assertDictEqual(
            stop_map,
            {
                "a": {0: 0, 1: 1},
            },
        )

    def test_combine_tau_steps_3(self):
        new_taus, start_map, stop_map = combine_tau_steps(
            {
                "a": [1 / 3, 2 / 3],
                "b": [1 / 3, 2 / 3],
            }
        )
        self.assertListAlmostEqual(new_taus, [1 / 3, 2 / 3])
        self.assertDictEqual(
            start_map,
            {
                "a": {0: 0, 1: 1},
                "b": {0: 0, 1: 1},
            },
        )
        self.assertDictEqual(
            stop_map,
            {
                "a": {0: 0, 1: 1},
                "b": {0: 0, 1: 1},
            },
        )

    def test_combine_tau_steps_4(self):
        new_taus, start_map, stop_map = combine_tau_steps(
            {
                "a": [0.5, 0.5],
                "b": [0.2, 0.4, 0.4],
                "c": [0.1, 0.7, 0.2],
                "d": [0.5, 0.5],
            }
        )
        self.assertListAlmostEqual(new_taus, [0.1, 0.1, 0.3, 0.1, 0.2, 0.2])
        self.assertDictEqual(
            start_map,
            {
                "a": {0: 0, 1: 3},
                "b": {0: 0, 1: 2, 2: 4},
                "c": {0: 0, 1: 1, 2: 5},
                "d": {0: 0, 1: 3},
            },
        )
        self.assertDictEqual(
            stop_map,
            {
                "a": {0: 2, 1: 5},
                "b": {0: 1, 1: 3, 2: 5},
                "c": {0: 0, 1: 4, 2: 5},
                "d": {0: 2, 1: 5},
            },
        )

    def test_remap_taus_1(self):
        class Clause1(MovementClause):
            leaves = TickIndex(0)
            returns = TickDelta(days=0, step=1)
            predicate = EveryDay()

            def evaluate(self, tick: Tick) -> NDArray[np.int64]:
                return np.array([])

        class Model1(MovementModel):
            steps = (1 / 3, 2 / 3)
            clauses = (Clause1(),)

        class Clause2(MovementClause):
            leaves = TickIndex(1)
            returns = TickDelta(days=0, step=1)
            predicate = EveryDay()

            def evaluate(self, tick: Tick) -> NDArray[np.int64]:
                return np.array([])

        class Model2(MovementModel):
            steps = (1 / 2, 1 / 2)
            clauses = (Clause2(),)

        new_mms = remap_taus([("a", Model1()), ("b", Model2())])

        new_taus = new_mms["a"].steps
        self.assertListAlmostEqual(new_taus, [1 / 3, 1 / 6, 1 / 2])
        self.assertEqual(len(new_mms), 2)

        new_mm1 = new_mms["a"]
        self.assertEqual(new_mm1.clauses[0].leaves.step, 0)
        self.assertEqual(new_mm1.clauses[0].returns.step, 2)

        new_mm2 = new_mms["b"]
        self.assertEqual(new_mm2.clauses[0].leaves.step, 2)
        self.assertEqual(new_mm2.clauses[0].returns.step, 2)


class RumeTest(EpymorphTestCase):
    def test_create_monostrata_1(self):
        # A single-strata RUME uses the IPM without modification.
        sir = Sir()
        centroids = mm_library["centroids"]()
        # Make sure centroids has the tau steps we will expect later...
        self.assertListAlmostEqual(centroids.steps, [1 / 3, 2 / 3])

        rume = SingleStrataRume.build(
            ipm=sir,
            mm=centroids,
            init=init.NoInfection(),
            scope=StateScope.in_states(["04", "35"], year=2020),
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
            rume.compartment_mask[DEFAULT_STRATA],
            [True, True, True],
        )
        assert_array_equal(
            rume.compartment_mobility[DEFAULT_STRATA],
            [True, True, True],
        )

    def test_create_multistrata_1(self):
        # Test a multi-strata model.

        sir = Sir()
        no = mm_library["no"]()
        # Make sure 'no' has the tau steps we will expect later...
        self.assertListAlmostEqual(no.steps, [1.0])

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
            scope=StateScope.in_states(["04", "35"], year=2020),
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
            rume.compartment_mask["aaa"],
            [True, True, True, False, False, False],
        )
        assert_array_equal(
            rume.compartment_mask["bbb"],
            [False, False, False, True, True, True],
        )
        assert_array_equal(
            rume.compartment_mobility["aaa"],
            [True, True, True, False, False, False],
        )
        assert_array_equal(
            rume.compartment_mobility["bbb"],
            [False, False, False, True, True, True],
        )

        # NOTE: these tests will break if someone alters the MM or Init definition;
        # even just the comments
        self.assertDictEqual(
            rume.requirements,
            {
                AbsoluteName("gpm:aaa", "ipm", "beta"): AttributeDef(
                    "beta", float, Shapes.TxN
                ),
                AbsoluteName("gpm:aaa", "ipm", "gamma"): AttributeDef(
                    "gamma", float, Shapes.TxN
                ),
                AbsoluteName("gpm:bbb", "ipm", "beta"): AttributeDef(
                    "beta", float, Shapes.TxN
                ),
                AbsoluteName("gpm:bbb", "ipm", "gamma"): AttributeDef(
                    "gamma", float, Shapes.TxN
                ),
                AbsoluteName("gpm:aaa", "init", "population"): AttributeDef(
                    "population",
                    int,
                    Shapes.N,
                    comment="The population at each geo node.",
                ),
                AbsoluteName("gpm:bbb", "init", "population"): AttributeDef(
                    "population",
                    int,
                    Shapes.N,
                    comment="The population at each geo node.",
                ),
            },
        )

    def test_create_multistrata_2(self):
        # Test special case: a multi-strata model but with only one strata.

        sir = Sir()
        centroids = mm_library["centroids"]()
        # Make sure centroids has the tau steps we will expect later...
        self.assertListAlmostEqual(centroids.steps, [1 / 3, 2 / 3])

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
            scope=StateScope.in_states(["04", "35"], year=2020),
            time_frame=TimeFrame.of("2021-01-01", 180),
            params={},
        )

        self.assertEqual(rume.dim.compartments, 3)
        self.assertEqual(rume.dim.events, 2)
        self.assertEqual(rume.dim.days, 180)
        self.assertEqual(rume.dim.ticks, 360)
        self.assertListAlmostEqual(rume.dim.tau_step_lengths, [1 / 3, 2 / 3])
        self.assertEqual(rume.dim.nodes, 2)

        # NOTE: these tests will break if someone alters the MM or Init definition;
        # even just the comments
        self.assertDictEqual(
            rume.requirements,
            {
                AbsoluteName("gpm:aaa", "ipm", "beta"): AttributeDef(
                    "beta", float, Shapes.TxN
                ),
                AbsoluteName("gpm:aaa", "ipm", "gamma"): AttributeDef(
                    "gamma", float, Shapes.TxN
                ),
                AbsoluteName("gpm:aaa", "mm", "population"): AttributeDef(
                    "population",
                    int,
                    Shapes.N,
                    comment="The total population at each node.",
                ),
                AbsoluteName("gpm:aaa", "mm", "centroid"): AttributeDef(
                    "centroid",
                    (("longitude", float), ("latitude", float)),
                    Shapes.N,
                    comment=(
                        "The centroids for each node as (longitude, latitude) tuples."
                    ),
                ),
                AbsoluteName("gpm:aaa", "mm", "phi"): AttributeDef(
                    "phi",
                    float,
                    Shapes.S,
                    comment="Influences the distance that movers tend to travel.",
                    default_value=40.0,
                ),
                AbsoluteName("gpm:aaa", "mm", "commuter_proportion"): AttributeDef(
                    "commuter_proportion",
                    float,
                    Shapes.S,
                    default_value=0.1,
                    comment="The proportion of the total population that commutes.",
                ),
                AbsoluteName("gpm:aaa", "init", "population"): AttributeDef(
                    "population",
                    int,
                    Shapes.N,
                    comment="The population at each geo node.",
                ),
            },
        )
