"""
This module provides the ParticleFilter class for running a particle filter on epidemiological data.
The filter estimates dynamic and static parameters through particle propagation and resampling.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from epymorph.parameter_fitting.filters.base_filters import BaseFilter
from epymorph.parameter_fitting.utils.epymorph_simulation import EpymorphSimulation
from epymorph.parameter_fitting.utils.particle_initializer import ParticleInitializer
from epymorph.parameter_fitting.utils.resampler import WeightsResampling
from epymorph.parameter_fitting.utils.utils import Utils


class ParticleFilter_multiplenodes(BaseFilter):
    """
    A class to run the particle filter for estimating parameters in epidemiological models.

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
        self.observations_quantiles = {}
        self.observations_values = {}
        self.beta_quantiles = []
        self.beta_values = []
        self.rng = np.random.default_rng(1)
        self.utils = Utils()

    def propagate_particles(
        self,
        particles: List[Tuple[np.ndarray, Dict[str, float]]],
        sim: Any,
        simulation: EpymorphSimulation,
        date: str,
        duration: int,
        is_sum: bool,
        model_link: str,
        observation: int,
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
        for x, observations in particles:
            propagated_x, state = simulation.propagate(
                x, observations, sim, date, duration, is_sum, model_link
            )

            if propagated_x is None:
                return None, None

            new_observations = {}

            for param, val in observations.items():
                if param == "beta":
                    if pd.isna(observation):
                        new_observations[param] = val
                    else:
                        new_observations[param] = np.exp(
                            np.random.normal(np.log(val), 0.1 * np.sqrt(duration))
                        )
                else:
                    new_observations[param] = val

            propagated_particles.append((propagated_x, new_observations))
            states.append((state, new_observations))
            # new_infections.append(infections)

        static_params = ["gamma", "eta"]
        if static_params in list(observations.keys()):
            log_static_params = np.log(
                np.array(
                    [
                        [obs[param] for param in static_params]
                        for _, obs in propagated_particles
                    ]
                )
            )
            weights = np.ones(len(particles)) / len(particles)
            log_mean = np.average(log_static_params, axis=0, weights=weights)
            cov = np.cov(log_static_params.T, aweights=weights)
            a = np.sqrt(1 - (0.5**2))

            for i in range(len(particles)):
                if len(static_params) == 1:
                    new_statics = np.exp(
                        self.rng.normal(
                            a * log_static_params[i] + (1 - a) * log_mean,
                            (0.5**2) * np.sqrt(cov),
                        )
                    )
                else:
                    new_statics = np.exp(
                        self.rng.multivariate_normal(
                            a * log_static_params[i] + (1 - a) * log_mean,
                            (0.5**2) * cov,
                        )
                    )
                for j, static in enumerate(new_statics):
                    propagated_particles[i][1][static_params[j]] = static

        return propagated_particles, states

    def run(
        self,
        rume: Dict[str, Any],
        likelihood_fn: Any,
        p_estimates: Dict[str, Tuple[float, float]],
        observations: Dict[str, Any],
        model_link: Any,
    ):
        """
        Runs the particle filter to estimate parameters.

        Args:
            rume (Dict[str, Any]): Model parameters including population size, seed size, static parameters, and geographical information.
            likelihood_fn (Any): Likelihood function.
            p_estimates (Dict[str, Tuple[float, float]]): Dynamic parameters and their ranges.
            observations (Dict[str, Any]): Observations including dates and cases.
            model_link (Any): Model linking function.

        Returns:
            Dict[str, List[float]]: Estimated parameter values.
        """
        dates = observations["Date"]
        data = observations["Cases"]
        is_sum = observations["is_sum"]
        num_observations = len(data)

        initializer = ParticleInitializer(self.num_particles, rume, p_estimates)
        particles = initializer.initialize_particles()
        simulation = EpymorphSimulation(rume, dates[0].strftime("%Y-%m-%d"))
        weights_resampling = WeightsResampling(
            self.num_particles, rume, likelihood_fn, model_link
        )
        sim = simulation.initialize_simulation(particles[0])

        for key in p_estimates.keys():
            self.observations_quantiles[key] = []
            self.observations_values[key] = []

        estimated_new_infections = []

        t = 0

        model_data = []

        params = rume["static_params"]

        while t < num_observations:
            print("Iteration: ", t)
            n = 1
            # print("particles[0] = ", particles[0][0].shape)
            if t > 0:
                # while pd.isna(data[t]):
                #     t += 1
                #     n += 1
                duration = (dates[t] - dates[t - n]).days
                # duration = 1
                # print("duration=",duration)
                propagated_particles, states = self.propagate_particles(
                    particles,
                    sim,
                    simulation,
                    dates[t].strftime("%Y-%m-%d"),
                    duration,
                    is_sum,
                    model_link,
                    data[t],
                )
                # print("propagate")
            else:
                # while pd.isna(data[t]):
                #     t += 1
                propagated_particles = states = particles

            if propagated_particles is None:
                print(
                    f"Warning: Data passed cannot be cummulative over the given model link {model_link} as it can overcome the total population\nPlease try a different model link or is_sum False"
                )
                break

            # model_data.append(
            #     np.mean([params['eta'] * particle[0][0][1] for particle in states]))

            total_propagated_particles = [pp[0][:15] for pp in states]

            # print("total_propagated_particles = ",total_propagated_particles[:5])
            # print("total_propagated_particles = ",total_propagated_particles[0].shape)
            # print("\n\n\n")

            total_propagated_particles = [
                [np.sum(pp, axis=0).reshape(1, 4)] for pp in total_propagated_particles
            ]

            model_data.append(
                np.mean(
                    [
                        params["eta"] * particle[0][0][1]
                        for particle in total_propagated_particles
                    ]
                )
            )

            # total_propagated_particles = np.array(total_propagated_particles)

            # print("total_propagated_particles = ",total_propagated_particles[0])
            # print("total_propagated_particles = ",total_propagated_particles[0][0].shape)

            if pd.isna(data[t]):
                particles = total_propagated_particles.copy()

            else:
                new_weights = weights_resampling.compute_weights(
                    data[t], total_propagated_particles
                )

                if np.any(np.isnan(new_weights)):
                    raise ValueError("NaN values found in computed weights.")

                particles = weights_resampling.resample_particles(
                    propagated_particles, new_weights
                )

                # print("particles[0] resampled = ", particles[0][0].shape)

            self.beta_quantiles.append(
                self.utils.quantiles([particle[1]["beta"] for particle in particles])
            )
            self.beta_values.append(
                np.mean([particle[1]["beta"] for particle in particles])
            )

            for key in p_estimates.keys():
                self.observations_quantiles[key].append(
                    self.utils.quantiles([particle[1][key] for particle in particles])
                )
                self.observations_values[key].append(
                    np.mean([particle[1][key] for particle in particles])
                )

            t += 1

        self.utils.save_data(self.observations_quantiles, self.observations_values)

        out = {"beta_quantiles": self.beta_quantiles, "beta_values": self.beta_values}

        return self.observations_values, out, estimated_new_infections, model_data
