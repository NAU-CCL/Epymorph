# pylint: disable=missing-docstring
import numpy as np

from epymorph.engine.world_hypercube import HypercubeWorld
from epymorph.simulation import SimDimensions, SimDType
from epymorph.test import EpymorphTestCase


class TestHypercubeWorld(EpymorphTestCase):

    _dim = SimDimensions.build([0.5, 0.5], 30, 4, 2, 2)
    _initials = np.array([[100, 25], [200, 75], [300, 125], [400, 175]], dtype=SimDType)

    def test_construct(self):
        world = HypercubeWorld.from_initials(self._dim, self._initials)
        self.assertNpEqual(
            self._initials,
            world.get_local_array()
        )

    def test_travel_1(self):
        world = HypercubeWorld.from_initials(self._dim, self._initials)
        self.assertEqual(world.time_frontier, 1)

        # Move 10 of c0 and 5 of c1 to each other node.
        # (Which totals 30 and 15.)
        travelers = np.stack([
            np.array([
                [0, 10, 10, 10],
                [10, 0, 10, 10],
                [10, 10, 0, 10],
                [10, 10, 10, 0],
            ], dtype=SimDType),
            np.array([
                [0, 5, 5, 5],
                [5, 0, 5, 5],
                [5, 5, 0, 5],
                [5, 5, 5, 0],
            ], dtype=SimDType),
        ], axis=2)

        world.apply_travel(travelers, 1)

        # We should have removed that many individuals from home
        self.assertNpEqual(
            world.get_local_array(),
            self._initials - [30, 15]
        )

        # And added those individuals at time index 2
        self.assertNpEqual(
            world.ledger[2, :, :, :],
            travelers
        )

        # And our frontier should be time index 3
        self.assertEqual(world.time_frontier, 3)

    def test_travel_2(self):
        world = HypercubeWorld.from_initials(self._dim, self._initials)
        self.assertEqual(world.time_frontier, 1)

        # Move 10 of c0 and 5 of c1 to each other node.
        # (Which totals 30 and 15.)
        travelers = np.stack([
            np.array([
                [0, 10, 10, 10],
                [10, 0, 10, 10],
                [10, 10, 0, 10],
                [10, 10, 10, 0],
            ], dtype=SimDType),
            np.array([
                [0, 5, 5, 5],
                [5, 0, 5, 5],
                [5, 5, 0, 5],
                [5, 5, 5, 0],
            ], dtype=SimDType),
        ], axis=2)

        # Apply movement a few times and see what happens
        world.apply_travel(travelers, 1)
        world.apply_travel(travelers, 1)
        world.apply_travel(travelers, 2)

        # We should have removed that many individuals from home
        self.assertNpEqual(
            world.get_local_array(),
            self._initials - [90, 45]
        )

        # And added individuals at time index 2 and 3
        self.assertNpEqual(
            world.ledger[2, :, :, :],
            travelers * 2
        )
        self.assertNpEqual(
            world.ledger[3, :, :, :],
            travelers
        )

        # And our frontier should be time index 4
        self.assertEqual(world.time_frontier, 4)

    def test_travel_3(self):
        world = HypercubeWorld.from_initials(self._dim, self._initials)
        self.assertEqual(world.time_frontier, 1)

        # Now try with non-samey numbers of movers
        travelers = np.stack([
            np.array([
                [0, 11, 12, 13],  # 36
                [14, 0, 15, 16],  # 45
                [17, 18, 0, 19],  # 54
                [20, 21, 22, 0],  # 63
            ], dtype=SimDType),
            np.array([
                [0, 1, 2, 3],  # 6
                [4, 0, 5, 6],  # 15
                [7, 8, 0, 9],  # 24
                [1, 5, 7, 0],  # 13
            ], dtype=SimDType),
        ], axis=2)

        world.apply_travel(travelers, 1)

        # We should have removed that many individuals from home
        self.assertNpEqual(
            world.get_local_array(),
            self._initials - np.array([[36, 6], [45, 15], [54, 24], [63, 13]])
        )

        # And added those individuals at time index 2
        self.assertNpEqual(
            world.ledger[2, :, :, :],
            travelers
        )

        # And our frontier should be time index 3
        self.assertEqual(world.time_frontier, 3)

        # Check the cohorts
        self.assertNpEqual(
            world.get_cohort_array(2),
            np.array([
                # t == 0
                [0, 0],  # these are the current "home" folks
                [0, 0],
                [246, 101],
                [0, 0],
                # t == 1
                [0, 0],  # nobody returns at time index 1
                [0, 0],
                [0, 0],
                [0, 0],
                # t == 2
                [12, 2],  # moved here from 0
                [15, 5],  # moved here from 1
                [0, 0],
                [22, 7],  # moved here from 3
            ], dtype=SimDType)
        )
