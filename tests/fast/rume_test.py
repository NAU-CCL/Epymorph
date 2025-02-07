# ruff: noqa: PT009,PT027
import unittest
from typing import Sequence

import numpy as np
from numpy.testing import assert_array_equal
from numpy.typing import NDArray

from epymorph.attribute import AbsoluteName, AttributeDef
from epymorph.compartment_model import CompartmentModel, compartment, edge
from epymorph.data.mm.centroids import Centroids
from epymorph.data.mm.no import No
from epymorph.data_shape import Shapes
from epymorph.geography.us_census import StateScope
from epymorph.initializer import NoInfection, SingleLocation
from epymorph.movement_model import EveryDay, MovementClause, MovementModel
from epymorph.rume import (
    GPM,
    MultiStrataRUME,
    SingleStrataRUME,
    combine_tau_steps,
    remap_taus,
)
from epymorph.simulation import NEVER, Tick, TickDelta, TickIndex
from epymorph.strata import DEFAULT_STRATA
from epymorph.time import TimeFrame


def assert_list_almost_equal(
    self,
    list1: Sequence[float],
    list2: Sequence[float],
    msg: str | None = None,
) -> None:
    """Check that two lists of numbers are approximately equal."""
    self.assertEqual(len(list1), len(list2), msg=msg)
    for a, b in zip(list1, list2):
        self.assertAlmostEqual(a, b, msg=msg)


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
        [S, I, R] = symbols.all_compartments  # noqa: N806
        [beta, gamma] = symbols.all_requirements
        return [
            edge(S, I, rate=beta * S * I),
            edge(I, R, rate=gamma * I),
        ]


class CombineMmTest(unittest.TestCase):
    def test_combine_tau_steps_1(self):
        new_taus, start_map, stop_map = combine_tau_steps(
            {
                "a": [1 / 3, 2 / 3],
                "b": [1 / 2, 1 / 2],
            }
        )
        assert_list_almost_equal(self, new_taus, [1 / 3, 1 / 6, 1 / 2])
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
        assert_list_almost_equal(self, new_taus, [1 / 3, 2 / 3])
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
        assert_list_almost_equal(self, new_taus, [1 / 3, 2 / 3])
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
        assert_list_almost_equal(self, new_taus, [0.1, 0.1, 0.3, 0.1, 0.2, 0.2])
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
        assert_list_almost_equal(self, new_taus, [1 / 3, 1 / 6, 1 / 2])
        self.assertEqual(len(new_mms), 2)

        new_mm1 = new_mms["a"]
        self.assertEqual(new_mm1.clauses[0].leaves.step, 0)
        self.assertEqual(new_mm1.clauses[0].returns.step, 2)

        new_mm2 = new_mms["b"]
        self.assertEqual(new_mm2.clauses[0].leaves.step, 2)
        self.assertEqual(new_mm2.clauses[0].returns.step, 2)

    def test_remap_taus_2(self):
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
            returns = NEVER
            predicate = EveryDay()

            def evaluate(self, tick: Tick) -> NDArray[np.int64]:
                return np.array([])

        class Model2(MovementModel):
            steps = (1 / 2, 1 / 2)
            clauses = (Clause2(),)

        new_mms = remap_taus([("a", Model1()), ("b", Model2())])

        new_taus = new_mms["a"].steps
        assert_list_almost_equal(self, new_taus, [1 / 3, 1 / 6, 1 / 2])
        self.assertEqual(len(new_mms), 2)

        new_mm1 = new_mms["a"]
        self.assertEqual(new_mm1.clauses[0].leaves.step, 0)
        self.assertEqual(new_mm1.clauses[0].returns.step, 2)

        new_mm2 = new_mms["b"]
        self.assertEqual(new_mm2.clauses[0].leaves.step, 2)
        self.assertEqual(new_mm2.clauses[0].returns.step, -1)


class RumeTest(unittest.TestCase):
    def test_create_monostrata_1(self):
        # A single-strata RUME uses the IPM without modification.
        sir = Sir()
        centroids = Centroids()
        # Make sure centroids has the tau steps we will expect later...
        assert_list_almost_equal(self, centroids.steps, [1 / 3, 2 / 3])

        rume = SingleStrataRUME.build(
            ipm=sir,
            mm=centroids,
            init=NoInfection(),
            scope=StateScope.in_states(["04", "35"], year=2020),
            time_frame=TimeFrame.of("2021-01-01", 180),
            params={},
        )
        self.assertIs(sir, rume.ipm)

        self.assertEqual(rume.num_ticks, 360)
        assert_list_almost_equal(self, rume.tau_step_lengths, [1 / 3, 2 / 3])

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
        no = No()
        # Make sure 'no' has the tau steps we will expect later...
        assert_list_almost_equal(self, no.steps, [1.0])

        rume = MultiStrataRUME.build(
            strata=[
                GPM(
                    name="aaa",
                    ipm=sir,
                    mm=no,
                    init=SingleLocation(location=0, seed_size=100),
                ),
                GPM(
                    name="bbb",
                    ipm=sir,
                    mm=no,
                    init=SingleLocation(location=0, seed_size=100),
                ),
            ],
            meta_requirements=[],
            meta_edges=lambda _: [],
            scope=StateScope.in_states(["04", "35"], year=2020),
            time_frame=TimeFrame.of("2021-01-01", 180),
            params={},
        )

        self.assertEqual(rume.num_ticks, 180)
        assert_list_almost_equal(self, rume.tau_step_lengths, [1.0])

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
        centroids = Centroids()
        # Make sure centroids has the tau steps we will expect later...
        assert_list_almost_equal(self, centroids.steps, [1 / 3, 2 / 3])

        rume = MultiStrataRUME.build(
            strata=[
                GPM(
                    name="aaa",
                    ipm=sir,
                    mm=centroids,
                    init=NoInfection(),
                ),
            ],
            meta_requirements=[],
            meta_edges=lambda _: [],
            scope=StateScope.in_states(["04", "35"], year=2020),
            time_frame=TimeFrame.of("2021-01-01", 180),
            params={},
        )

        self.assertEqual(rume.num_ticks, 360)
        assert_list_almost_equal(self, rume.tau_step_lengths, [1 / 3, 2 / 3])

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
                    Shapes.Scalar,
                    comment="Influences the distance that movers tend to travel.",
                    default_value=40.0,
                ),
                AbsoluteName("gpm:aaa", "mm", "commuter_proportion"): AttributeDef(
                    "commuter_proportion",
                    float,
                    Shapes.Scalar,
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
