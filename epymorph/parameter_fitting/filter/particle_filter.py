import time
from typing import Any, Dict, List, Tuple, Type

import numpy as np
import pandas as pd

from epymorph.parameter_fitting.dynamics import (
    GeometricBrownianMotion,
)
from epymorph.parameter_fitting.filter.base_filter import BaseFilter
from epymorph.parameter_fitting.filter.particle import Particle
from epymorph.parameter_fitting.likelihood import Likelihood
from epymorph.parameter_fitting.output import ParticleFilterOutput
from epymorph.parameter_fitting.perturbation import Calvetti
from epymorph.parameter_fitting.utils import utils
from epymorph.parameter_fitting.utils.epymorph_simulation import EpymorphSimulation
from epymorph.parameter_fitting.utils.observations import ModelLink
from epymorph.parameter_fitting.utils.parameter_estimation import EstimateParameters
from epymorph.parameter_fitting.utils.params_perturb import Perturb
from epymorph.parameter_fitting.utils.particle_initializer import ParticleInitializer
from epymorph.parameter_fitting.utils.resampler import WeightsResampling
from epymorph.rume import RUME


class ParticleFilter(BaseFilter):
    """
    A class to run the particle filter for estimating parameters.

    Attributes
    ----------
    num_particles : int
        Number of particles.
    param_quantiles : Dict[str, List[np.ndarray]]
        Quantiles of parameters over time.
    param_values : Dict[str, List[np.ndarray]]
        Mean values of parameters over time.
    resampler: Type[WeightsResampling]
        The resampling method to use.
    rng : np.random.Generator
        Random number generator for simulations and resampling.
    """

    def __init__(
        self, num_particles: int, resampler: Type[WeightsResampling] = WeightsResampling
    ) -> None:
        """
        Initializes the ParticleFilter with the given number of particles.

        Parameters
        ----------
        num_particles : int
            Number of particles for the particle filter.
        resampler : Type[WeightsResampling], optional
            The resampler to use in the particle filter.
        """
        self.num_particles = num_particles
        self.param_quantiles = {}
        self.param_values = {}
        self.resampler = resampler

    def propagate_particles(
        self,
        particles: List[Particle],
        rume: RUME,
        simulation: EpymorphSimulation,
        date: str,
        duration: int,
        model_link: ModelLink,
        params_space: Dict[str, EstimateParameters],
        rng: np.random.Generator,
    ) -> Tuple[List[Particle], List[np.ndarray]]:
        """
        Propagates particles through the simulation model.

        Parameters
        ----------
        particles : List[Particle]
            The particles which represent the current estimate of the filter.
        rume : Rume
            Model parameters including population size and geographical information.
        simulation : EpymorphSimulation
            The object which propagates the particles.
        date : str
            Current date.
        duration : int
            Duration of propagation in days.
        model_link : ModelLink
            Information used to match the model output with the observations.
        params_space : Dict[str, EstimateParameters]
            The parameters to be estimated and the methods used to estimate them.

        Returns
        -------
        Tuple
            - A list of propagated Particle objects with updated observations.
            - A list of expected observations for each particle after propagation.
        """
        propagated_particles = []
        expected_observations = []

        # Initialize perturbation handler
        params_perturb = Perturb(duration)

        # Propagate each particle through the model
        for particle in particles:
            # Use the particle's state and parameters for propagation
            new_state, observation = simulation.propagate(
                particle.state,
                particle.parameters,
                rume,
                date,
                duration,
                model_link,
                rng,
            )

            # Update the parameters using their dynamics
            new_parameters = {}
            for param, val in particle.parameters.items():
                dynamics = params_space[param].dynamics
                if isinstance(dynamics, GeometricBrownianMotion):
                    new_parameters[param] = params_perturb.gbm(
                        val, dynamics.volatility, rng
                    )
                else:
                    new_parameters[param] = val

            # Create a new particle with the propagated state and updated parameters
            propagated_particles.append(Particle(new_state, new_parameters))

            expected_observations.append(observation)

        return propagated_particles, expected_observations

    def run(
        self,
        rume: RUME,
        likelihood_fn: Likelihood,
        params_space: Dict[str, EstimateParameters],
        model_link: ModelLink,
        dates: Any,
        cases: List[np.ndarray],
        rng: np.random.Generator,
    ) -> ParticleFilterOutput:
        """
        Runs the particle filter to estimate parameters.

        Parameters
        ----------
        rume : Rume
            Model parameters, including population size and geographical information.
        likelihood_fn : Likelihood
            The likelihood function to use in the resampling.
        params_space : Dict[str, EstimateParameters]
            The parameters to estimate and the methods to estimate them.
        model_link : ModelLink
            Link to the model used for simulations.
        index : int
            Index of the parameter to estimate.
        dates : Any
            Dates for which observations are available.
        cases : List[int]
            Observed case data over time.

        Returns
        -------
        ParticleFilterOutput
            The result of the particle filter containing parameter estimates, quantiles,
            and model data.
        """
        start_time = time.time()
        dates = pd.to_datetime(dates)
        data = np.array(cases)

        # Ensure data is 2D for compatibility
        if len(data.shape) == 1:
            data = data[:, np.newaxis]  # Reshape to 2D array (N, 1)

        print("Running Particle Filter simulation")  # noqa: T201
        print(f"• {dates[0]} to {dates[-1]} ({rume.time_frame.duration_days} days)")  # noqa: T201
        print(f"• {self.num_particles} particles")  # noqa: T201

        num_observations = len(data)

        # Initialize the particles, simulation, and resampling tools
        initializer = ParticleInitializer(self.num_particles, rume, params_space)
        particles = initializer.initialize_particles(rng)
        simulation = EpymorphSimulation(rume, dates[0])
        weights_resampling = self.resampler(
            self.num_particles,
            likelihood_fn,
        )

        # Prepare containers for storing results
        for key in params_space.keys():
            self.param_quantiles[key] = []
            self.param_values[key] = []

        model_data = []

        # Iterate through each time step and perform filtering
        for t in range(num_observations):
            n = 1  # Number of days to look back for the previous observation
            if t > 0:
                duration = (dates[t] - dates[t - n]).days
            else:
                duration = 1

            # Propagate particles and update their states
            propagated_particles, expected_observations = self.propagate_particles(
                particles,
                rume,
                simulation,
                dates[t].strftime("%Y-%m-%d"),
                duration,
                model_link,
                params_space,
                rng,
            )

            # Append model data (mean of particle states) for this time step
            model_data.append(
                np.mean(
                    [obs for obs in expected_observations],
                    axis=0,
                ).astype(int)  # Ensure the final mean is also an integer
            )

            if np.all(np.isnan(data[t, ...])):
                particles = propagated_particles.copy()

            else:
                # Now pass all observations for the current time step
                # Pass the entire observation (all columns for that time step)
                new_weights = weights_resampling.compute_weights(
                    data[t, ...],  # This will pass all data for the current time step
                    expected_observations,
                )

                if np.any(np.isnan(new_weights)):
                    raise ValueError("NaN values found in computed weights.")
                particles = weights_resampling.resample_particles(
                    propagated_particles, new_weights, rng
                )

            for param in particles[0].parameters.keys():
                perturbation = params_space[param].perturbation
                if isinstance(perturbation, Calvetti):
                    param_vals = np.array(
                        [particle.parameters[param] for particle in particles]
                    )
                    param_mean = np.mean(np.log(param_vals), axis=0)
                    param_cov = np.cov(np.log(param_vals), rowvar=False)
                    a = perturbation.a
                    h = np.sqrt(1 - a**2)
                    if len(param_cov.shape) < 2:
                        param_cov = np.broadcast_to(param_cov, shape=(1, 1))
                    rvs = rng.multivariate_normal(
                        (1 - a) * param_mean, h**2 * param_cov, size=len(particles)
                    )
                    for i in range(len(particles)):
                        particles[i].parameters[param] = np.exp(
                            a * np.log(particles[i].parameters[param]) + rvs[i, ...]
                        )

            # Collect parameter values for quantiles and means
            key_values = {key: [] for key in self.param_quantiles.keys()}
            for particle in particles:
                for key in key_values.keys():
                    if key in particle.parameters:
                        key_values[key].append(particle.parameters[key])

            # Store quantiles and means for each parameter
            for key, values in key_values.items():
                if values:
                    self.param_quantiles[key].append(utils.quantiles(np.array(values)))
                    self.param_values[key].append(np.mean(values))

        parameters_estimated = list(self.param_quantiles.keys())
        # Calculate total runtime
        total_runtime = time.time() - start_time
        print(f"\nSimulation completed in {total_runtime:.2f}s")  # noqa: T201
        print(f"\nParameters estimated: {parameters_estimated}")  # noqa: T201

        # Prepare the output object
        out = ParticleFilterOutput(
            self.num_particles,
            parameters_estimated,
            str(rume.time_frame.duration_days) + " days",
            self.param_quantiles,
            self.param_values,
            true_data=np.array(data),
            model_data=np.array(model_data),
        )

        return out
