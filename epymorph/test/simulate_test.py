# pylint: disable=missing-docstring
import unittest
from functools import partial
from textwrap import dedent

import numpy as np

from epymorph import *
from epymorph.compartment_model import (CompartmentModel, compartment,
                                        create_model, create_symbols, edge,
                                        param)
from epymorph.error import IpmSimLessThanZeroException, IpmSimNaNException
from epymorph.geo.spec import NO_DURATION, AttribDef, StaticGeoSpec
from epymorph.geo.static import StaticGeo
from epymorph.initializer import single_location


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

        err_msg = '''
              Less than zero rate detected. When providing or defining ipm parameters, ensure that
              they will not result in a negative rate. Note: this can often happen unintentionally
              if a function is given as a parameter.

              Showing current ipm params
              infection_duration: 0.25
              immunity_duration: -0.01
              humidity: 0.01003
              '''
        err_msg = dedent(err_msg)

        self.assertEqual(str(e.exception), err_msg)

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
                    param('beta', shape=Shapes.TxN),  # infectivity
                    # progression from infected to recovered
                    param('gamma', shape=Shapes.TxN),
                    # progression from recovered to susceptible
                    param('xi', shape=Shapes.TxN)
                ])

            [S, I, R] = symbols.compartment_symbols
            [β, γ, ξ] = symbols.attribute_symbols

            # LOOK!
            # N is NOT protected by Max(1, ...) here
            # This isn't necessary for Case 1, but is necessary for Case 2.
            N = S + I + R

            return create_model(
                symbols=symbols,
                transitions=[
                    edge(S, I, rate=β * S * I / N),
                    edge(I, R, rate=γ * I),
                    edge(R, S, rate=ξ * R)
                ])

        ipm = load_ipm()

        my_geo = StaticGeo(
            spec=StaticGeoSpec(
                attributes=[
                    AttribDef('label', dtype=str, shape=Shapes.N),
                    AttribDef('population', dtype=str, shape=Shapes.N),
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
                'xi': 1 / 90,
            },
            time_frame=TimeFrame.of("2015-01-01", 150),
            initializer=partial(single_location, location=1, seed_size=5),
            rng=default_rng(1)
        )

        with self.assertRaises(IpmSimNaNException) as e:
            sim.run()

        err_msg = '''
              NaN (not a number) rate detected. This is often the result of a divide by zero error.
              When constructing the IPM, ensure that no edge transitions can result in division by zero
              This commonly occurs when defining an S->I edge that is (some rate / sum of the compartments)
              To fix this, change the edge to define the S->I edge as (some rate / Max(1/sum of the the compartments))
              See examples of this in the provided example ipm definitions in the data/ipms folder.

              Showing current events
              S->I: I*S*beta/(I + R + S)
              I->R: I*gamma
              R->S: R*xi
              '''
        err_msg = dedent(err_msg)

        self.assertEqual(str(e.exception), err_msg)
