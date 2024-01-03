# pylint: disable=missing-docstring
import unittest
from functools import partial

import numpy as np

from epymorph import *
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
