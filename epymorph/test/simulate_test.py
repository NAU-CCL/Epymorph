# pylint: disable=missing-docstring
import unittest
from functools import partial

import numpy as np

from epymorph import *
from epymorph.compartment_model import (CompartmentModel, compartment,
                                        create_model, create_symbols, edge)
from epymorph.error import (IpmSimInvalidProbsException,
                            IpmSimLessThanZeroException, IpmSimNaNException,
                            MmSimException)
from epymorph.geo.spec import NO_DURATION, StaticGeoSpec
from epymorph.geo.static import StaticGeo
from epymorph.initializer import single_location
from epymorph.simulation import geo_attrib, params_attrib


class SimulateTest(unittest.TestCase):
    """
    Testing that simulations seem to actually work.
    This is more of an integration test, but it's quick enough and critical
    for epymorph's correctness.
    """

    def test_pei(self):
        sim = StandardSimulation(
            geo=geo_library['pei'](),
            ipm=ipm_library['pei'](),
            mm=mm_library['pei'](),
            params={
                'infection_duration': 1 / 4,
                'immunity_duration': 1 / 90,
                'move_control': 0.9,
                'theta': 0.1,
            },
            time_frame=TimeFrame.of("2015-01-01", 10),
            initializer=partial(single_location, location=0, seed_size=10_000),
            rng=default_rng(42)
        )

        out1 = sim.run()

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

        out2 = sim.run()

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

    def test_less_than_zero_err(self):
        """Test exception handling for a negative rate value due to a negative parameter"""
        sim = StandardSimulation(
            geo=geo_library['pei'](),
            ipm=ipm_library['pei'](),
            mm=mm_library['pei'](),
            params={
                'infection_duration': 1 / 4,
                'immunity_duration': -1 / 100,  # notice the negative parameter
                'move_control': 0.9,
                'theta': 0.1,
            },
            time_frame=TimeFrame.of("2015-01-01", 10),
            initializer=partial(single_location, location=0, seed_size=10_000),
            rng=default_rng(42)
        )

        with self.assertRaises(IpmSimLessThanZeroException) as e:
            sim.run()

        err_msg = str(e.exception)

        self.assertIn("Less than zero rate detected", err_msg)
        self.assertIn("Showing current Node : Timestep", err_msg)
        self.assertIn("S: ", err_msg)
        self.assertIn("I: ", err_msg)
        self.assertIn("R: ", err_msg)
        self.assertIn("infection_duration: 0.25", err_msg)
        self.assertIn("immunity_duration: -0.01", err_msg)
        self.assertIn("humidity: 0.01003", err_msg)

    def test_divide_by_zero_err(self):
        """Test exception handling for a divide by zero (NaN) error"""
        def load_ipm() -> CompartmentModel:
            """Load the 'sirs' IPM."""
            symbols = create_symbols(
                compartments=[
                    compartment('S'),
                    compartment('I'),
                    compartment('R'),
                ],
                attributes=[
                    params_attrib('beta', dtype=float, shape=Shapes.TxN),  # infectivity
                    # progression from infected to recovered
                    params_attrib('gamma', dtype=float, shape=Shapes.TxN),
                    # progression from recovered to susceptible
                    params_attrib('xi', dtype=float, shape=Shapes.TxN)
                ])

            [S, I, R] = symbols.compartment_symbols
            [β, γ, ξ] = symbols.attribute_symbols

            # N is NOT protected by Max(1, ...) here
            N = S + I + R

            return create_model(
                symbols=symbols,
                transitions=[
                    edge(S, I, rate=β * S * I / N),
                    edge(I, R, rate=γ * I),
                    edge(R, S, rate=ξ * R)
                ])

        my_geo = StaticGeo(
            spec=StaticGeoSpec(
                attributes=[
                    geo_attrib('label', dtype=str, shape=Shapes.N),
                    geo_attrib('population', dtype=str, shape=Shapes.N),
                ],
                time_period=NO_DURATION,
            ),
            values={
                'label': np.array(['a', 'b', 'c']),
                'population': np.array([0, 10, 20], dtype=np.int64),
            },
        )
        sim = StandardSimulation(
            geo=my_geo,
            ipm=load_ipm(),
            mm=mm_library['no'](),
            params={
                'phi': 40.0,
                'beta': 0.4,
                'gamma': 1 / 5,
                'xi': 1 / 100,
            },
            time_frame=TimeFrame.of("2015-01-01", 150),
            initializer=partial(single_location, location=1, seed_size=5),
            rng=default_rng(1)
        )

        with self.assertRaises(IpmSimNaNException) as e:
            sim.run()

        err_msg = str(e.exception)

        self.assertIn("NaN (not a number) rate detected", err_msg)
        self.assertIn("Showing current Node : Timestep", err_msg)
        self.assertIn("S: 0", err_msg)
        self.assertIn("I: 0", err_msg)
        self.assertIn("R: 0", err_msg)
        self.assertIn("beta: 0.4", err_msg)
        self.assertIn("gamma: 0.2", err_msg)
        self.assertIn("xi: 0.01", err_msg)
        self.assertIn("S->I: I*S*beta/(I + R + S)", err_msg)

    def test_negative_probs_error(self):
        """Test for handling negative probability error"""
        sim = StandardSimulation(
            geo=geo_library['pei'](),
            ipm=ipm_library['sirh'](),
            mm=mm_library['no'](),
            params={
                'beta': 0.4,
                'gamma': 1 / 5,
                'xi': 1 / 100,
                'hospitalization_prob': -1 / 5,
                'hospitalization_duration': 15
            },
            time_frame=TimeFrame.of("2015-01-01", 150),
            initializer=partial(single_location, location=1, seed_size=5),
            rng=default_rng(1)
        )

        with self.assertRaises(IpmSimInvalidProbsException) as e:
            sim.run()

        err_msg = str(e.exception)

        self.assertIn("Invalid probabilities for fork definition detected.", err_msg)
        self.assertIn("Showing current Node : Timestep", err_msg)
        self.assertIn("S: ", err_msg)
        self.assertIn("I: ", err_msg)
        self.assertIn("R: ", err_msg)
        self.assertIn("beta: 0.4", err_msg)
        self.assertIn("gamma: 0.2", err_msg)
        self.assertIn("xi: 0.01", err_msg)
        self.assertIn("hospitalization_prob: -0.2", err_msg)
        self.assertIn("hospitalization_duration: 15", err_msg)

        self.assertIn("I->(H, R): I*gamma", err_msg)
        self.assertIn(
            "Probabilities: hospitalization_prob, 1 - hospitalization_prob", err_msg)

    def test_mm_clause_error(self):
        """Test for handling invalid movement model clause application"""

        sim = StandardSimulation(
            geo=geo_library['pei'](),
            ipm=ipm_library['pei'](),
            mm=mm_library['pei'](),
            params={
                'infection_duration': 40.0,
                'immunity_duration': 0.4,
                'humidity': 20.2,
                'move_control': 0.4,
                'theta': -5
            },
            time_frame=TimeFrame.of("2015-01-01", 150),
            initializer=partial(single_location, location=1, seed_size=5),
            rng=default_rng(1)
        )

        with self.assertRaises(MmSimException) as e:
            sim.run()

        err_msg = str(e.exception)

        self.assertIn(
            "Error from applying clause 'dispersers': see exception trace", err_msg)
