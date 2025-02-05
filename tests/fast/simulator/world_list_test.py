# ruff: noqa: PT009,PT027
import unittest
from datetime import date

import numpy as np
from numpy.typing import ArrayLike

from epymorph.data_type import SimDType
from epymorph.simulation import Tick
from epymorph.simulator.world_list import Cohort, ListWorld


def _ndarray_signature(arr):
    return f"{type(arr).__name__}: dtype({arr.dtype}) shape{repr(arr.shape)}"


def assert_np_equal(self, a1: ArrayLike, a2: ArrayLike, msg: str | None = None) -> None:
    """Check that two numpy ArrayLikes are equal."""
    if not np.array_equal(a1, a2):
        if msg is None:
            if not isinstance(a1, np.ndarray):
                a1 = np.asarray(a1)
            if not isinstance(a2, np.ndarray):
                a2 = np.asarray(a2)
            sig1 = _ndarray_signature(a1)
            sig2 = _ndarray_signature(a2)
            msg = f"""\
arrays not equal
- a1: {sig1}
{str(a1)}
- a2: {sig2}
{str(a2)}"""
        self.fail(msg)


class TestCohort(unittest.TestCase):
    def test_can_merge(self):
        c1 = Cohort(np.array([100]), 0, ListWorld.HOME_TICK)
        c2 = Cohort(np.array([200]), 0, ListWorld.HOME_TICK)
        self.assertTrue(c1.can_merge_with(c1))
        self.assertTrue(c2.can_merge_with(c2))
        self.assertTrue(c1.can_merge_with(c2))
        self.assertTrue(c2.can_merge_with(c1))

    def test_can_not_merge(self):
        # if return tick varies -- no merge
        c1 = Cohort(np.array([100]), 0, 1)
        c2 = Cohort(np.array([200]), 0, 2)
        self.assertFalse(c1.can_merge_with(c2))
        # if return location varies -- no merge
        c3 = Cohort(np.array([300]), 1, 1)
        self.assertFalse(c1.can_merge_with(c3))
        # if both vary -- no merge
        self.assertFalse(c2.can_merge_with(c3))

    def test_merge(self):
        c1 = Cohort(np.array([100, 33]), 13, 42)
        c2 = Cohort(np.array([200, 66]), 13, 42)
        c1.merge_from(c2)
        # c1 (only compartment value) has changed...
        assert_np_equal(self, c1.compartments, [300, 99])
        self.assertEqual(c1.return_location, 13)
        self.assertEqual(c1.return_tick, 42)
        # c2 remains unchanged...
        assert_np_equal(self, c2.compartments, [200, 66])
        self.assertEqual(c2.return_location, 13)
        self.assertEqual(c2.return_tick, 42)


HOME_TICK = ListWorld.HOME_TICK


class TestListWorld(unittest.TestCase):
    def assert_world(
        self,
        world_a: list[list[Cohort]],
        world_b: list[list[Cohort]],
    ) -> None:
        if len(world_a) != len(world_b):
            msg = (
                "Worlds contained a different number of locations: "
                f"{len(world_a)} vs. {len(world_b)}"
            )
            self.fail(msg)

        for i, (loc_a, loc_b) in enumerate(zip(world_a, world_b)):
            if len(loc_a) != len(loc_b):
                msg = (
                    f"Location {i} contained a different number of cohorts: "
                    f"{len(loc_a)} vs. {len(loc_b)}"
                )
                self.fail(msg)

            for j, (coh_a, coh_b) in enumerate(zip(loc_a, loc_b)):
                if coh_a != coh_b:
                    msg = f"Location {i}, Cohort {j} were not equal:\n{coh_a}\n{coh_b}"
                    self.fail(msg)

    def test_normalize_01(self):
        # simple normalization: just two cohorts
        actual = ListWorld(
            [
                [
                    Cohort(np.array([100]), 0, HOME_TICK),
                    Cohort(np.array([200]), 0, HOME_TICK),
                ]
            ]
        )

        expected = [[Cohort(np.array([300]), 0, HOME_TICK)]]

        actual.normalize()

        self.assert_world(actual.locations, expected)

    def test_normalize_02(self):
        # more complex normalization: some cohorts to combine, not already sorted
        actual = ListWorld(
            [
                [
                    Cohort(np.array([200]), 0, HOME_TICK),
                    Cohort(np.array([75]), 2, 2),
                    Cohort(np.array([100]), 0, HOME_TICK),
                    Cohort(np.array([50]), 1, 2),
                ]
            ]
        )

        expected = [
            [
                Cohort(np.array([300]), 0, HOME_TICK),
                Cohort(np.array([50]), 1, 2),
                Cohort(np.array([75]), 2, 2),
            ]
        ]

        actual.normalize()

        self.assert_world(actual.locations, expected)

    def test_normalize_03(self):
        # normalization of a bunch of mergeable cohorts
        actual = ListWorld(
            [
                [
                    Cohort(np.array([100]), 0, HOME_TICK),
                    Cohort(np.array([100]), 0, HOME_TICK),
                    Cohort(np.array([100]), 0, HOME_TICK),
                    Cohort(np.array([100]), 0, HOME_TICK),
                    Cohort(np.array([100]), 0, HOME_TICK),
                ]
            ]
        )

        expected = [
            [
                Cohort(np.array([500]), 0, HOME_TICK),
            ]
        ]

        actual.normalize()

        self.assert_world(actual.locations, expected)

    def test_cohort_array(self):
        world = ListWorld(
            [
                [
                    Cohort(np.array([3]), 0, HOME_TICK),
                    Cohort(np.array([5]), 1, 1),
                    Cohort(np.array([7]), 1, 2),
                ],
                [
                    Cohort(np.array([13]), 1, HOME_TICK),
                    Cohort(np.array([15]), 0, 1),
                    Cohort(np.array([17]), 0, 2),
                ],
            ]
        )
        assert_np_equal(
            self,
            world.get_cohort_array(0),
            np.array([[3], [5], [7]], dtype=SimDType),
        )
        assert_np_equal(
            self,
            world.get_cohort_array(1),
            np.array([[13], [15], [17]], dtype=SimDType),
        )

    def test_local_array(self):
        world = ListWorld(
            [
                [
                    Cohort(np.array([3]), 0, HOME_TICK),
                    Cohort(np.array([5]), 1, 1),
                    Cohort(np.array([7]), 1, 2),
                ],
                [
                    Cohort(np.array([13]), 1, HOME_TICK),
                    Cohort(np.array([15]), 0, 1),
                    Cohort(np.array([17]), 0, 2),
                ],
            ]
        )
        assert_np_equal(
            self,
            world.get_local_array(),
            np.array([[3], [13]], dtype=SimDType),
        )

    def test_apply_delta(self):
        world = ListWorld(
            [
                [
                    Cohort(np.array([100, 10]), 0, HOME_TICK),
                    Cohort(np.array([200, 20]), 1, 1),
                    Cohort(np.array([300, 30]), 1, 2),
                ],
                [
                    Cohort(np.array([100, 10]), 0, HOME_TICK),
                    Cohort(np.array([200, 20]), 1, 1),
                    Cohort(np.array([300, 30]), 1, 2),
                ],
            ]
        )

        world.apply_cohort_delta(
            1, np.array([[-10, 10], [-20, 20], [-30, 30]], dtype=SimDType)
        )

        self.assert_world(
            world.locations,
            [
                [
                    Cohort(np.array([100, 10]), 0, HOME_TICK),
                    Cohort(np.array([200, 20]), 1, 1),
                    Cohort(np.array([300, 30]), 1, 2),
                ],
                [
                    Cohort(np.array([90, 20]), 0, HOME_TICK),
                    Cohort(np.array([180, 40]), 1, 1),
                    Cohort(np.array([270, 60]), 1, 2),
                ],
            ],
        )

    def test_travel_01(self):
        # Test simple travel.
        world = ListWorld(
            [
                [
                    Cohort(np.array([100]), 0, HOME_TICK),
                    Cohort(np.array([22]), 1, 1),
                ],
                [
                    Cohort(np.array([200]), 1, HOME_TICK),
                    Cohort(np.array([11]), 0, 1),
                ],
            ]
        )

        world.apply_travel(
            travelers=np.array(
                [
                    [[0], [19]],
                    [[29], [0]],
                ],
                dtype=SimDType,
            ),
            return_tick=2,
        )

        # Movement cohorts maintain their identity.
        self.assert_world(
            world.locations,
            [
                [
                    Cohort(np.array([81]), 0, HOME_TICK),
                    Cohort(np.array([22]), 1, 1),
                    Cohort(np.array([29]), 1, 2),
                ],
                [
                    Cohort(np.array([171]), 1, HOME_TICK),
                    Cohort(np.array([11]), 0, 1),
                    Cohort(np.array([19]), 0, 2),
                ],
            ],
        )

    def test_travel_02(self):
        # Test travel with a "never" return tick.
        world = ListWorld(
            [
                [
                    Cohort(np.array([100]), 0, HOME_TICK),
                    Cohort(np.array([22]), 1, 1),
                ],
                [
                    Cohort(np.array([200]), 1, HOME_TICK),
                    Cohort(np.array([11]), 0, 1),
                ],
            ]
        )

        world.apply_travel(
            travelers=np.array(
                [
                    [[0], [19]],
                    [[29], [0]],
                ],
                dtype=SimDType,
            ),
            return_tick=-1,
        )

        # Movement cohorts merge with locals.
        self.assert_world(
            world.locations,
            [
                [
                    Cohort(np.array([110]), 0, HOME_TICK),
                    Cohort(np.array([22]), 1, 1),
                ],
                [
                    Cohort(np.array([190]), 1, HOME_TICK),
                    Cohort(np.array([11]), 0, 1),
                ],
            ],
        )

    def test_return(self):
        world = ListWorld(
            [
                [
                    Cohort(np.array([81]), 0, HOME_TICK),
                    Cohort(np.array([22]), 1, 1),
                    Cohort(np.array([29]), 1, 2),
                ],
                [
                    Cohort(np.array([171]), 1, HOME_TICK),
                    Cohort(np.array([11]), 0, 1),
                    Cohort(np.array([19]), 0, 2),
                ],
            ]
        )

        world.apply_return(Tick(1, 1, date(2023, 1, 1), 0, 1.0), return_stats=False)

        self.assert_world(
            world.locations,
            [
                [
                    Cohort(np.array([92]), 0, HOME_TICK),
                    Cohort(np.array([29]), 1, 2),
                ],
                [
                    Cohort(np.array([193]), 1, HOME_TICK),
                    Cohort(np.array([19]), 0, 2),
                ],
            ],
        )
