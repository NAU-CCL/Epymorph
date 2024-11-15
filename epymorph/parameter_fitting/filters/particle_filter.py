"""
This module provides the ParticleFilter class for running a particle filter
on epidemiological data.
The filter estimates dynamic and static parameters through particle propagation
and resampling.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from epymorph.parameter_fitting.dynamics.dynamics import GeometricBrownianMotion
from epymorph.parameter_fitting.filters.base_filters import BaseFilter
from epymorph.parameter_fitting.output import ParticleFilterOutput
from epymorph.parameter_fitting.utils import utils
from epymorph.parameter_fitting.utils.epymorph_simulation import EpymorphSimulation
from epymorph.parameter_fitting.utils.parameter_estimation import EstimateParameters
from epymorph.parameter_fitting.utils.ParamsPerturb import Perturb
from epymorph.parameter_fitting.utils.particle_initializer import ParticleInitializer
from epymorph.parameter_fitting.utils.resampler import WeightsResampling


class ParticleFilter(BaseFilter):
    """
    A class to run the particle filter for estimating parameters in epidemiological
    models.

    Attributes:
        num_particles (int): Number of particles.
        observations_quantiles (Dict[str, List[float]]): Quantiles of observations.
        observations_values (Dict[str, List[float]]): Mean values of observations.
        rng (np.random.Generator): Random number generator.
        utils (Utils): Utility functions for quantiles and data saving.
    """

    def __init__(self, num_particles: int, resampler=WeightsResampling) -> None:
        """
        Initializes the ParticleFilter with the given number of particles.

        Args:
            num_particles (int): Number of particles.
        """
        self.num_particles = num_particles
        self.param_quantiles = {}
        self.param_values = {}
        self.rng = np.random.default_rng()
        self.resampler = resampler

    def propagate_particles(
        self,
        particles: List[Tuple[np.ndarray, Dict[str, float]]],
        rume: Any,
        simulation: EpymorphSimulation,
        date: str,
        duration: int,
        is_sum: bool,
        model_link: str,
        observation: int,
        params_space,
    ):
        """
        Propagates particles through the simulation model.

        Args:
            particles (List[Tuple[np.ndarray, Dict[str, float]]]): List of particles.
            sim (Any): Simulation object.
            simulation (EpymorphSimulation): Epymorph simulation object.
            date (str): Current date in simulation.
            duration (int): Duration of propagation.
            is_sum (bool): Whether to sum the beta values over the duration.

        Returns:
            List[Tuple[np.ndarray, Dict[str, float]]]: List of propagated particles.
        """
        propagated_particles = []
        states = []

        params_perturb = Perturb(duration)
        for compartments, parameters in particles:
            propagated_x, state = simulation.propagate(
                compartments, parameters, rume, date, duration, is_sum, model_link
            )

            new_parameters = {}

            for param, val in parameters.items():
                dynamics = params_space[param].dynamics
                if isinstance(dynamics, GeometricBrownianMotion):
                    new_parameters[param] = val
                    new_parameters[param] = params_perturb.gbm(val, dynamics.volatility)
                else:
                    new_parameters[param] = val

            propagated_particles.append((propagated_x, new_parameters))
            states.append((state, 0))

        return propagated_particles, states

    def run(
        self,
        rume: Any,
        likelihood_fn: Any,
        params_space: Dict[str, EstimateParameters],
        model_link: Any,
        index: int,
        dates: Any,
        cases: Any,
    ) -> Any:
        """
        Runs the particle filter to estimate parameters.

        Args:
            rume (Dict[str, Any]): Model parameters including population size,
            seed size, static parameters, and geographical information.
            likelihood_fn (Any): Likelihood function.
            p_estimates (Dict[str, EstimateParameters): Dynamic parameters and their
            ranges.
            observations (Dict[str, Any]): Observations including dates and cases.
            model_link (Any): Model linking function.

        Returns:
            Dict[str, List[float]]: Estimated parameter values.
        """
        dates = pd.to_datetime(dates)
        data = cases
        if len(data.shape) == 1:
            data = data[:, np.newaxis]
        is_sum = True
        num_observations = len(data)

        initializer = ParticleInitializer(self.num_particles, rume, params_space)
        particles = initializer.initialize_particles()
        simulation = EpymorphSimulation(rume, dates[0])
        weights_resampling = self.resampler(
            self.num_particles, rume, likelihood_fn, model_link, index
        )

        for key in params_space.keys():
            self.param_quantiles[key] = []
            self.param_values[key] = []

        t = 0

        model_data = []

        for t in tqdm(range(num_observations)):
            n = 1
            if t > 0:
                duration = (dates[t] - dates[t - n]).days
                propagated_particles, states = self.propagate_particles(
                    particles,
                    rume,
                    simulation,
                    dates[t].strftime("%Y-%m-%d"),
                    duration,
                    is_sum,
                    model_link,
                    0,  # The observation is unused.
                    params_space,
                )
            else:
                propagated_particles = states = particles

            model_data.append(
                np.mean(
                    [np.array(particle[0])[:, index] for particle in states], axis=0
                )
            )

            total_propagated_particles = propagated_particles

            if np.all(np.isnan(data[t, ...])):
                particles = total_propagated_particles.copy()

            else:
                new_weights = weights_resampling.compute_weights(data[t, ...], states)

                if np.any(np.isnan(new_weights)):
                    raise ValueError("NaN values found in computed weights.")

                particles = weights_resampling.resample_particles(
                    total_propagated_particles, new_weights
                )

            key_values = {key: [] for key in self.param_quantiles.keys()}

            for particle in particles:
                for key in key_values.keys():
                    if key in particle[1]:
                        key_values[key].append(particle[1][key])

            for key, values in key_values.items():
                if values:
                    self.param_quantiles[key].append(utils.quantiles(np.array(values)))
                    self.param_values[key].append(np.mean(values))

            t += 1

        out = ParticleFilterOutput(
            self.num_particles,
            str(rume.time_frame.duration_days) + " days ",
            self.param_quantiles,
            self.param_values,
            np.array(data),
            np.array(model_data),
        )

        return out
