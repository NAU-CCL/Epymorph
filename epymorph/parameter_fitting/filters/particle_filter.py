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
from epymorph.parameter_fitting.utils.ParameterEstimation import EstimateParameters
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
        beta_quantiles (List[float]): Quantiles of beta values.
        beta_values (List[float]): Mean values of beta.
        rng (np.random.Generator): Random number generator.
        utils (Utils): Utility functions for quantiles and data saving.
    """

    def __init__(self, num_particles: int) -> None:
        """
        Initializes the ParticleFilter with the given number of particles.

        Args:
            num_particles (int): Number of particles.
        """
        self.num_particles = num_particles
        self.param_quantiles = {}
        self.param_values = {}
        self.beta_quantiles = []
        self.beta_values = []
        self.rng = np.random.default_rng()
        # self.utils = Utils()

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
        # ) -> List[Tuple[np.ndarray, Dict[str, float]]]:
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
        for x, observations in particles:
            propagated_x, state = simulation.propagate(
                x, observations, rume, date, duration, is_sum, model_link
            )

            # if propagated_x is None:
            #     return None, None

            new_observations = {}

            for param, val in observations.items():
                dynamics = params_space[param].dynamics
                if isinstance(dynamics, GeometricBrownianMotion):
                    if pd.isna(observation):
                        new_observations[param] = val
                    else:
                        new_observations[param] = params_perturb.gbm(
                            val, dynamics.volatility
                        )
                        # new_observations[param] = np.exp(
                        #     np.random.normal(np.log(val), 0.1 * np.sqrt(duration))
                        # )
                else:
                    new_observations[param] = val

            propagated_particles.append((propagated_x, new_observations))
            states.append((state, 0))
            # new_infections.append(infections)

        # static_params = ["gamma", "eta"]
        # if static_params in list(observations.keys()):
        #     log_static_params = np.log(
        #         np.array(
        #             [
        #                 [obs[param] for param in static_params]
        #                 for _, obs in propagated_particles
        #             ]
        #         )
        #     )
        #     weights = np.ones(len(particles)) / len(particles)
        #     log_mean = np.average(log_static_params, axis=0, weights=weights)
        #     cov = np.cov(log_static_params.T, aweights=weights)
        #     a = np.sqrt(1 - (0.5**2))

        #     for i in range(len(particles)):
        #         if len(static_params) == 1:
        #              new_statics = np.exp(
        #                 self.rng.normal(
        #                     a * log_static_params[i] + (1 - a) * log_mean,
        #                     (0.5**2) * np.sqrt(cov),
        #                 )
        #             )
        #         else:
        #             new_statics = np.exp(
        #                 self.rng.multivariate_normal(
        #                     a * log_static_params[i] + (1 - a) * log_mean,
        #                     (0.5**2) * cov,
        #                 )
        #             )
        #         for j, static in enumerate(new_statics):
        #             propagated_particles[i][1][static_params[j]] = static

        return propagated_particles, states

    def run(
        self,
        # **kwargs: Any,
        rume: Any,
        likelihood_fn: Any,
        params_space: Dict[str, EstimateParameters],
        # observations: Dict[str, Any],
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
        # is_sum = observations["is_sum"]
        is_sum = True
        num_observations = len(data)

        initializer = ParticleInitializer(self.num_particles, rume, params_space)
        particles = initializer.initialize_particles()
        simulation = EpymorphSimulation(rume, dates[0])  # .strftime("%Y-%m-%d"))
        weights_resampling = WeightsResampling(
            self.num_particles, rume, likelihood_fn, model_link, index
        )

        # sim = simulation.initialize_simulation(particles[0])
        for key in params_space.keys():
            self.param_quantiles[key] = []
            self.param_values[key] = []

        t = 0

        model_data = []

        # while t < num_observations:
        for t in tqdm(range(num_observations)):
            # print("Iteration: ", t)
            n = 1
            if t > 0:
                # while pd.isna(data[t]):
                #     t += 1
                #     n += 1
                duration = (dates[t] - dates[t - n]).days
                # duration = 1
                # print("duration=", duration)
                propagated_particles, states = self.propagate_particles(
                    particles,
                    rume,
                    simulation,
                    dates[t].strftime("%Y-%m-%d"),
                    duration,
                    is_sum,
                    model_link,
                    data[t],
                    params_space,
                )
                # print("propagate")
            else:
                # while pd.isna(data[t]):
                #     t += 1
                propagated_particles = states = particles

            model_data.append(np.mean([particle[0][0][1] for particle in states]))

            total_propagated_particles = [
                (np.sum(pp[0], axis=0).reshape(1, -1), {"beta": pp[1]["beta"]})
                for pp in propagated_particles
            ]

            if pd.isna(data[t]):
                particles = total_propagated_particles.copy()

            else:
                new_weights = weights_resampling.compute_weights(data[t], states)

                if np.any(np.isnan(new_weights)):
                    raise ValueError("NaN values found in computed weights.")

                particles = weights_resampling.resample_particles(
                    total_propagated_particles, new_weights
                )

            self.beta_quantiles.append(
                utils.quantiles([particle[1]["beta"] for particle in particles])
            )
            self.beta_values.append(
                np.mean([particle[1]["beta"] for particle in particles])
            )

            key_values = {key: [] for key in self.param_quantiles.keys()}

            for particle in particles:
                for key in key_values.keys():
                    if key in particle[1]:
                        key_values[key].append(particle[1][key])

            for key, values in key_values.items():
                if values:
                    self.param_quantiles[key].append(utils.quantiles(values))
                    self.param_values[key].append(np.mean(values))

            t += 1

        # utils.save_data(self.param_quantiles, True)
        # utils.save_data(self.param_values, False)

        # out = [{"beta_quantiles": self.beta_quantiles, "beta_values": self.beta_values}]

        # out = ["infection rate", self.beta_values]

        # model_data_df = pd.DataFrame(model_data)

        # model_data_df.to_csv(
        #     "./epymorph/parameter_fitting/data/model_data.csv", index=False
        # )
        out = ParticleFilterOutput(
            self.num_particles,
            str(rume.time_frame.duration_days) + " days ",
            self.param_quantiles,
            self.param_values,
            np.array(data),
            np.array(model_data),
        )

        return out
