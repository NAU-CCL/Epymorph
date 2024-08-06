# pylint: disable=missing-docstring
import unittest
from math import inf
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from epymorph import *
from epymorph.compartment_model import CompartmentModel, compartment, edge
from epymorph.error import (IpmSimInvalidProbsException,
                            IpmSimLessThanZeroException, IpmSimNaNException,
                            MmSimException)
from epymorph.geography.scope import CustomScope
from epymorph.geography.us_census import StateScope
from epymorph.rume import SingleStrataRume
from epymorph.simulation import AttributeDef


class SimulateTest(unittest.TestCase):
    """
    Testing that simulations seem to actually work.
    This is more of an integration test, but it's quick enough and critical
    for epymorph's correctness.
    """

    def _pei_scope(self) -> StateScope:
        pei_states = ["FL", "GA", "MD", "NC", "SC", "VA"]
        return StateScope.in_states_by_code(pei_states, 2010)

    def _pei_geo(self) -> Mapping[str, NDArray]:
        # We don't want to use real ADRIOs here because they could fail
        # and cause these tests to spuriously fail.
        # So instead, hard-code some values. They don't need to be real.
        t = np.arange(start=0, stop=2 * np.pi, step=2 * np.pi / 365)
        return {
            "*::population": np.array([18811310, 9687653, 5773552, 9535483, 4625364, 8001024]),
            "*::humidity": np.array([
                0.005 + 0.005 * np.sin(t) for _ in range(6)
            ]).T,
            "*::commuters": np.array([
                [7993452, 13805, 2410, 2938, 1783, 3879],
                [15066, 4091461, 966, 6057, 20318, 2147],
                [949, 516, 2390255, 947, 91, 122688],
                [3005, 5730, 1872, 4121984, 38081, 29487],
                [1709, 23513, 630, 64872, 1890853, 1620],
                [1368, 1175, 68542, 16869, 577, 3567788],
            ]),
        }

    def test_pei(self):
        rume = SingleStrataRume.build(
            ipm=ipm_library['pei'](),
            mm=mm_library['pei'](),
            init=init.SingleLocation(location=0, seed_size=10_000),
            scope=self._pei_scope(),
            time_frame=TimeFrame.of("2015-01-01", 10),
            params={
                'ipm::infection_duration': 4,
                'ipm::immunity_duration': 90,
                'mm::move_control': 0.9,
                'mm::theta': 0.1,
                **self._pei_geo(),
            },
        )

        sim = BasicSimulator(rume)

        out1 = sim.run(rng_factory=default_rng(42))

        np.testing.assert_array_equal(
            out1.initial[:, 1],
            np.array([10_000, 0, 0, 0, 0, 0], dtype=SimDType),
            "Output should contain accurate initials."
        )

        self.assertGreater(out1.prevalence[:, :, 0].max(), 0,
                           "S prevalence should be greater than zero at some point in the sim.")
        self.assertGreater(out1.prevalence[:, :, 1].max(), 0,
                           "I prevalence should be greater than zero at some point in the sim.")
        self.assertGreater(out1.prevalence[:, :, 2].max(), 0,
                           "R prevalence should be greater than zero at some point in the sim.")
        self.assertGreater(out1.incidence[:, :, 0].max(), 0,
                           "S-to-I incidence should be greater than zero at some point in the sim.")
        self.assertGreater(out1.incidence[:, :, 1].max(), 0,
                           "I-to-R incidence should be greater than zero at some point in the sim.")
        self.assertGreater(out1.incidence[:, :, 2].max(), 0,
                           "R-to-S incidence should be greater than zero at some point in the sim.")

        self.assertGreaterEqual(out1.prevalence.min(), 0,
                                "Prevalence can never be less than zero.")
        self.assertGreaterEqual(out1.incidence.min(), 0,
                                "Incidence can never be less than zero.")

        out2 = sim.run(
            rng_factory=default_rng(42),
        )

        np.testing.assert_array_equal(
            out1.incidence,
            out2.incidence,
            "Running the sim twice with the same RNG should yield the same incidence."
        )

        np.testing.assert_array_equal(
            out1.prevalence,
            out2.prevalence,
            "Running the sim twice with the same RNG should yield the same prevalence."
        )

    def test_override_params(self):
        rume = SingleStrataRume.build(
            ipm=ipm_library['pei'](),
            mm=mm_library['pei'](),
            init=init.SingleLocation(location=0, seed_size=10_000),
            scope=self._pei_scope(),
            time_frame=TimeFrame.of("2015-01-01", 10),
            params={
                'ipm::infection_duration': 4,
                'ipm::immunity_duration': 90,
                'mm::move_control': 0.9,
                'mm::theta': 0.1,
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
            params={'ipm::immunity_duration': inf},
            rng_factory=rng_factory,
        )

        # We expect in the first result, some people do make the R->S transition,
        self.assertFalse(np.all(out1.incidence[:, 0, 2] == 0))
        # while in the second result, no one does.
        self.assertTrue(np.all(out2.incidence[:, 0, 2] == 0))

    def test_less_than_zero_err(self):
        """Test exception handling for a negative rate value due to a negative parameter"""
        rume = SingleStrataRume.build(
            ipm=ipm_library['pei'](),
            mm=mm_library['pei'](),
            init=init.SingleLocation(location=0, seed_size=10_000),
            scope=self._pei_scope(),
            time_frame=TimeFrame.of("2015-01-01", 10),
            params={
                'ipm::infection_duration': 4,
                'ipm::immunity_duration': -100,  # notice the negative parameter
                'mm::move_control': 0.9,
                'mm::theta': 0.1,
                **self._pei_geo(),
            },
        )

        sim = BasicSimulator(rume)

        with self.assertRaises(IpmSimLessThanZeroException) as e:
            sim.run(rng_factory=default_rng(42))

        err_msg = str(e.exception)
        self.assertIn("Less than zero rate detected", err_msg)
        self.assertIn("immunity_duration: -100.0", err_msg)

    def test_divide_by_zero_err(self):
        """Test exception handling for a divide by zero (NaN) error"""
        class Sirs(CompartmentModel):
            compartments = [
                compartment('S'),
                compartment('I'),
                compartment('R'),
            ]

            requirements = [
                AttributeDef('beta', type=float, shape=Shapes.TxN),
                AttributeDef('gamma', type=float, shape=Shapes.TxN),
                AttributeDef('xi', type=float, shape=Shapes.TxN),
            ]

            def edges(self, symbols):
                [S, I, R] = symbols.all_compartments
                [β, γ, ξ] = symbols.all_requirements

                # N is NOT protected by Max(1, ...) here
                N = S + I + R  # type: ignore

                return [
                    edge(S, I, rate=β * S * I / N),
                    edge(I, R, rate=γ * I),
                    edge(R, S, rate=ξ * R),
                ]

        rume = SingleStrataRume.build(
            ipm=Sirs(),
            mm=mm_library['no'](),
            init=init.SingleLocation(location=1, seed_size=5),
            scope=CustomScope(np.array(['a', 'b', 'c'])),
            time_frame=TimeFrame.of("2015-01-01", 150),
            params={
                '*::mm::phi': 40.0,
                '*::ipm::beta': 0.4,
                '*::ipm::gamma': 1 / 5,
                '*::ipm::xi': 1 / 100,
                '*::*::population': np.array([0, 10, 20], dtype=np.int64),
            },
        )

        sim = BasicSimulator(rume)
        with self.assertRaises(IpmSimNaNException) as e:
            sim.run(rng_factory=default_rng(1))

        err_msg = str(e.exception)
        self.assertIn("NaN (not a number) rate detected", err_msg)
        self.assertIn("S: 0", err_msg)
        self.assertIn("I: 0", err_msg)
        self.assertIn("R: 0", err_msg)
        self.assertIn("S->I: I*S*beta/(I + R + S)", err_msg)

    def test_negative_probs_error(self):
        """Test for handling negative probability error"""
        rume = SingleStrataRume.build(
            ipm=ipm_library['sirh'](),
            mm=mm_library['no'](),
            init=init.SingleLocation(location=1, seed_size=5),
            scope=self._pei_scope(),
            time_frame=TimeFrame.of("2015-01-01", 150),
            params={
                'beta': 0.4,
                'gamma': 1 / 5,
                'xi': 1 / 100,
                'hospitalization_prob': -1 / 5,
                'hospitalization_duration': 15,
                **self._pei_geo(),
            },
        )

        sim = BasicSimulator(rume)
        with self.assertRaises(IpmSimInvalidProbsException) as e:
            sim.run(rng_factory=default_rng(1))

        err_msg = str(e.exception)
        self.assertIn("Invalid probabilities for fork definition detected.", err_msg)
        self.assertIn("hospitalization_prob: -0.2", err_msg)
        self.assertIn("hospitalization_duration: 15", err_msg)
        self.assertIn("I->(H, R): I*gamma", err_msg)
        self.assertIn(
            "Probabilities: hospitalization_prob, 1 - hospitalization_prob", err_msg)

    def test_mm_clause_error(self):
        """Test for handling invalid movement model clause application"""
        rume = SingleStrataRume.build(
            ipm=ipm_library['pei'](),
            mm=mm_library['pei'](),
            init=init.SingleLocation(location=1, seed_size=5),
            scope=self._pei_scope(),
            time_frame=TimeFrame.of("2015-01-01", 150),
            params={
                'infection_duration': 40.0,
                'immunity_duration': 0.4,
                'humidity': 20.2,
                'move_control': 0.4,
                'theta': -5.0,
                **self._pei_geo(),
            },
        )

        sim = BasicSimulator(rume)
        with self.assertRaises(MmSimException) as e:
            sim.run(rng_factory=default_rng(1))

        err_msg = str(e.exception)

        self.assertIn(
            "Error from applying clause 'dispersers': see exception trace", err_msg)
