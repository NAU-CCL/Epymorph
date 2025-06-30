# ruff: noqa: PT009,PT027
import unittest
from math import inf
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from epymorph.attribute import AttributeDef
from epymorph.compartment_model import CompartmentModel, compartment, edge
from epymorph.data.ipm.pei import Pei as PeiIPM
from epymorph.data.ipm.sirh import SIRH
from epymorph.data.mm.no import No as NoMM
from epymorph.data.mm.pei import Pei as PeiMM
from epymorph.data_shape import Shapes
from epymorph.data_type import SimDType
from epymorph.error import (
    IPMSimInvalidForkError,
    IPMSimLessThanZeroError,
    IPMSimNaNError,
    MMSimError,
)
from epymorph.geography.custom import CustomScope
from epymorph.geography.us_census import StateScope
from epymorph.initializer import SingleLocation
from epymorph.rume import SingleStrataRUME
from epymorph.simulation import default_rng
from epymorph.simulator.basic.basic_simulator import BasicSimulator
from epymorph.time import TimeFrame


class SimulateTest(unittest.TestCase):
    """
    Testing that simulations seem to actually work.
    This is more of an integration test, but it's quick enough and critical
    for epymorph's correctness.
    """

    def _pei_scope(self) -> StateScope:
        pei_states = ["FL", "GA", "MD", "NC", "SC", "VA"]
        return StateScope.in_states(pei_states, 2010)

    def _pei_geo(self) -> Mapping[str, NDArray]:
        # We don't want to use real ADRIOs here because they could fail
        # and cause these tests to spuriously fail.
        # So instead, hard-code some values. They don't need to be real.
        t = np.arange(start=0, stop=2 * np.pi, step=2 * np.pi / 365)
        return {
            "*::population": np.array(
                [18811310, 9687653, 5773552, 9535483, 4625364, 8001024]
            ),
            "*::humidity": np.array([0.005 + 0.005 * np.sin(t) for _ in range(6)]).T,
            "*::commuters": np.array(
                [
                    [7993452, 13805, 2410, 2938, 1783, 3879],
                    [15066, 4091461, 966, 6057, 20318, 2147],
                    [949, 516, 2390255, 947, 91, 122688],
                    [3005, 5730, 1872, 4121984, 38081, 29487],
                    [1709, 23513, 630, 64872, 1890853, 1620],
                    [1368, 1175, 68542, 16869, 577, 3567788],
                ]
            ),
        }

    def test_pei(self):
        rume = SingleStrataRUME.build(
            ipm=PeiIPM(),
            mm=PeiMM(),
            init=SingleLocation(location=0, seed_size=10_000),
            scope=self._pei_scope(),
            time_frame=TimeFrame.of("2015-01-01", 10),
            params={
                "ipm::infection_duration": 4,
                "ipm::immunity_duration": 90,
                "mm::move_control": 0.9,
                "mm::theta": 0.1,
                **self._pei_geo(),
            },
        )

        sim = BasicSimulator(rume)

        out1 = sim.run(rng_factory=default_rng(42))

        np.testing.assert_array_equal(
            out1.initial[:, 1],
            np.array([10_000, 0, 0, 0, 0, 0], dtype=SimDType),
            "Output should contain accurate initials.",
        )

        self.assertGreater(
            out1.compartments[:, :, 0].max(),
            0,
            "S compartment should be greater than zero at some point in the sim.",
        )
        self.assertGreater(
            out1.compartments[:, :, 1].max(),
            0,
            "I compartment should be greater than zero at some point in the sim.",
        )
        self.assertGreater(
            out1.compartments[:, :, 2].max(),
            0,
            "R compartment should be greater than zero at some point in the sim.",
        )
        self.assertGreater(
            out1.events[:, :, 0].max(),
            0,
            "S-to-I event should be greater than zero at some point in the sim.",
        )
        self.assertGreater(
            out1.events[:, :, 1].max(),
            0,
            "I-to-R event should be greater than zero at some point in the sim.",
        )
        self.assertGreater(
            out1.events[:, :, 2].max(),
            0,
            "R-to-S event should be greater than zero at some point in the sim.",
        )

        self.assertGreaterEqual(
            out1.compartments.min(), 0, "Compartments can never be less than zero."
        )
        self.assertGreaterEqual(
            out1.events.min(), 0, "Events can never be less than zero."
        )

        out2 = sim.run(
            rng_factory=default_rng(42),
        )

        np.testing.assert_array_equal(
            out1.events,
            out2.events,
            "Running the sim twice with the same RNG should yield the same events.",
        )

        np.testing.assert_array_equal(
            out1.compartments,
            out2.compartments,
            (
                "Running the sim twice with the same RNG should yield the same "
                "compartments."
            ),
        )

    def test_override_params(self):
        rume = SingleStrataRUME.build(
            ipm=PeiIPM(),
            mm=PeiMM(),
            init=SingleLocation(location=0, seed_size=10_000),
            scope=self._pei_scope(),
            time_frame=TimeFrame.of("2015-01-01", 10),
            params={
                "ipm::infection_duration": 4,
                "ipm::immunity_duration": 90,
                "mm::move_control": 0.9,
                "mm::theta": 0.1,
                **self._pei_geo(),
            },
        )

        sim = BasicSimulator(rume)
        rng_factory = default_rng(42)

        # Run once with immunity_duration = 90
        out1 = sim.run(
            rng_factory=rng_factory,
        )
        # And again with immunity_duration = inf
        out2 = sim.run(
            params={"ipm::immunity_duration": inf},
            rng_factory=rng_factory,
        )

        # We expect in the first result, some people do make the R->S transition,
        self.assertFalse(np.all(out1.events[:, 0, 2] == 0))
        # while in the second result, no one does.
        self.assertTrue(np.all(out2.events[:, 0, 2] == 0))

    def test_less_than_zero_err(self):
        """
        Test exception handling for a negative rate value due to a negative parameter
        """
        rume = SingleStrataRUME.build(
            ipm=PeiIPM(),
            mm=PeiMM(),
            init=SingleLocation(location=0, seed_size=10_000),
            scope=self._pei_scope(),
            time_frame=TimeFrame.of("2015-01-01", 10),
            params={
                "ipm::infection_duration": 4,
                "ipm::immunity_duration": -100,  # notice the negative parameter
                "mm::move_control": 0.9,
                "mm::theta": 0.1,
                **self._pei_geo(),
            },
        )

        sim = BasicSimulator(rume)

        with self.assertRaises(IPMSimLessThanZeroError) as e:
            sim.run(rng_factory=default_rng(42))

        err_msg = str(e.exception)
        self.assertIn("less than zero", err_msg)
        self.assertIn("immunity_duration: -100.0", err_msg)

    def test_divide_by_zero_err(self):
        """Test exception handling for a divide by zero (NaN) error"""

        class Sirs(CompartmentModel):
            compartments = [
                compartment("S"),
                compartment("I"),
                compartment("R"),
            ]

            requirements = [
                AttributeDef("beta", type=float, shape=Shapes.TxN),
                AttributeDef("gamma", type=float, shape=Shapes.TxN),
                AttributeDef("xi", type=float, shape=Shapes.TxN),
            ]

            def edges(self, symbols):
                [S, I, R] = symbols.all_compartments  # noqa: N806
                [β, γ, ξ] = symbols.all_requirements

                # N is NOT protected by Max(1, ...) here
                N = S + I + R  # type: ignore

                return [
                    edge(S, I, rate=β * S * I / N),
                    edge(I, R, rate=γ * I),
                    edge(R, S, rate=ξ * R),
                ]

        rume = SingleStrataRUME.build(
            ipm=Sirs(),
            mm=NoMM(),
            init=SingleLocation(location=1, seed_size=5),
            scope=CustomScope(np.array(["a", "b", "c"])),
            time_frame=TimeFrame.of("2015-01-01", 150),
            params={
                "*::mm::phi": 40.0,
                "*::ipm::beta": 0.4,
                "*::ipm::gamma": 1 / 5,
                "*::ipm::xi": 1 / 100,
                "*::*::population": np.array([0, 10, 20], dtype=np.int64),
            },
        )

        sim = BasicSimulator(rume)
        with self.assertRaises(IPMSimNaNError) as e:
            sim.run(rng_factory=default_rng(1))

        err_msg = str(e.exception)
        self.assertIn("transition rate was NaN", err_msg)
        self.assertIn("S: 0", err_msg)
        self.assertIn("I: 0", err_msg)
        self.assertIn("R: 0", err_msg)
        self.assertIn("S → I: I*S*beta/(I + R + S)", err_msg)

    def test_negative_probs_error(self):
        """Test for handling negative probability error"""
        rume = SingleStrataRUME.build(
            ipm=SIRH(),
            mm=NoMM(),
            init=SingleLocation(location=1, seed_size=5),
            scope=self._pei_scope(),
            time_frame=TimeFrame.of("2015-01-01", 150),
            params={
                "beta": 0.4,
                "gamma": 1 / 5,
                "xi": 1 / 100,
                "hospitalization_prob": -1 / 5,
                "hospitalization_duration": 15,
                **self._pei_geo(),
            },
        )

        sim = BasicSimulator(rume)
        with self.assertRaises(IPMSimInvalidForkError) as e:
            sim.run(rng_factory=default_rng(1))

        err_msg = str(e.exception)
        self.assertIn("fork transition is invalid", err_msg)
        self.assertIn("hospitalization_prob: -0.2", err_msg)
        self.assertIn("hospitalization_duration: 15", err_msg)
        self.assertIn("I → (H,R): I*gamma", err_msg)
        self.assertIn(
            "Probabilities: hospitalization_prob, 1 - hospitalization_prob", err_msg
        )

    def test_mm_clause_error(self):
        """Test for handling invalid movement model clause application"""
        rume = SingleStrataRUME.build(
            ipm=PeiIPM(),
            mm=PeiMM(),
            init=SingleLocation(location=1, seed_size=5),
            scope=self._pei_scope(),
            time_frame=TimeFrame.of("2015-01-01", 150),
            params={
                "infection_duration": 40.0,
                "immunity_duration": 0.4,
                "humidity": 20.2,
                "move_control": 0.4,
                "theta": -5.0,
                **self._pei_geo(),
            },
        )

        sim = BasicSimulator(rume)
        with self.assertRaises(MMSimError) as e:
            sim.run(rng_factory=default_rng(1))

        err_msg = str(e.exception)

        self.assertIn(
            "Error from applying clause 'Dispersers': see exception trace", err_msg
        )
