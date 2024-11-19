import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from epymorph.parameter_fitting.dynamics.dynamics import GeometricBrownianMotion
from epymorph.parameter_fitting.filters.base_filters import BaseFilter
from epymorph.parameter_fitting.filters.particle import Particle
from epymorph.parameter_fitting.output import ParticleFilterOutput
from epymorph.parameter_fitting.utils import utils
from epymorph.parameter_fitting.utils.epymorph_simulation import EpymorphSimulation
from epymorph.parameter_fitting.utils.parameter_estimation import EstimateParameters
from epymorph.parameter_fitting.utils.ParamsPerturb import Perturb
from epymorph.parameter_fitting.utils.particle_initializer import ParticleInitializer
from epymorph.parameter_fitting.utils.resampler import WeightsResampling


class ParticleFilter(BaseFilter):
    """
    A class to run the particle filter for estimating parameters in epidemiological models.

    Attributes:
        num_particles (int): Number of particles.
        param_quantiles (Dict[str, List[np.ndarray]]): Quantiles of parameters over time.
        param_values (Dict[str, List[np.ndarray]]): Mean values of parameters over time.
        beta_quantiles (List[float]): Quantiles of beta values.
        beta_values (List[float]): Mean values of beta.
        rng (np.random.Generator): Random number generator for simulations.
    """

    def __init__(self, num_particles: int, resampler=WeightsResampling) -> None:
        """
        Initializes the ParticleFilter with the given number of particles.

        Args:
            num_particles (int): Number of particles for the particle filter.
        """
        self.num_particles = num_particles
        self.param_quantiles = {}  # Stores quantiles for each parameter
        self.param_values = {}  # Stores mean values for each parameter
        self.rng = (
            np.random.default_rng()
        )  # Random number generator for particle simulations
        self.resampler = resampler

    def propagate_particles(
        self,
        particles: List[Particle],
        rume: Any,
        simulation: EpymorphSimulation,
        date: str,
        duration: int,
        is_sum: bool,
        model_link: str,
        observation: int,
        params_space: Dict[str, EstimateParameters],
    ) -> List[Particle]:
        """
        Propagates particles through the simulation model.

        Args:
            particles (List[Particle]): List of Particle objects.
            rume (Any): Model parameters including population size and geographical
            information.
            simulation (EpymorphSimulation): The simulation object that propagates the
            particles.
            date (str): Current date in simulation format.
            duration (int): Duration of propagation.
            is_sum (bool): Whether to sum the beta values over the duration.
            model_link (str): Link to the model to use for prediction.
            observation (int): Observation for the current time step.
            params_space (Dict[str, EstimateParameters]): Parameter space for the model.

        Returns:
            Tuple:
                - A list of propagated Particle objects with updated observations.
                - A list of states for each particle after propagation.
        """
        propagated_particles = []

        # Initialize perturbation handler
        params_perturb = Perturb(duration)

        # Propagate each particle through the model
        for particle in particles:
            # Use the particle's state and parameters for propagation
            propagated_state, events_state = simulation.propagate(
                particle.state,
                particle.parameters,
                rume,
                date,
                duration,
                is_sum,
                model_link,
            )

            # Update the parameters using their dynamics
            new_parameters = {}
            for param, val in particle.parameters.items():
                dynamics = params_space[param].dynamics
                if isinstance(dynamics, GeometricBrownianMotion):
                    new_parameters[param] = val
                    new_parameters[param] = params_perturb.gbm(val, dynamics.volatility)
                else:
                    new_parameters[param] = val

            # Create a new particle with the propagated state and updated parameters
            propagated_particles.append(
                Particle(propagated_state, new_parameters, events_state)
            )

        return propagated_particles

    def run(
        self,
        rume: Any,
        likelihood_fn: Any,
        params_space: Dict[str, EstimateParameters],
        model_link: Any,
        index: int,
        dates: Any,
        cases: List[np.ndarray],
    ) -> ParticleFilterOutput:
        """
        Runs the particle filter to estimate parameters.

        Args:
            rume (Any): Model parameters, including population size and
            geographical information.
            likelihood_fn (Any): The likelihood function to use in the resampling.
            params_space (Dict[str, EstimateParameters]): Dynamic parameters and
            their ranges.
            model_link (Any): Link to the model used for simulations.
            index (int): Index of the parameter to estimate.
            dates (List[str]): List of dates for which observations are available.
            cases (List[int]): Observed case data over time.

        Returns:
            ParticleFilterOutput: The result of the particle filter containing
            parameter estimates,
                quantiles, and model data.
        """
        start_time = time.time()
        dates = pd.to_datetime(dates)
        data = np.array(cases)

        # Ensure data is 2D for compatibility
        if len(data.shape) == 1:
            data = data[:, np.newaxis]  # Reshape to 2D array (N, 1)

        print("Running Particle Filter simulation")
        print(f"• {dates[0]} to {dates[-1]} ({rume.time_frame.duration_days} days)")
        print(f"• {self.num_particles} particles")

        is_sum = True  # Flag for summing the beta values
        num_observations = len(data)

        # Initialize the particles, simulation, and resampling tools
        initializer = ParticleInitializer(self.num_particles, rume, params_space)
        particles = initializer.initialize_particles()
        simulation = EpymorphSimulation(rume, dates[0])
        weights_resampling = self.resampler(
            self.num_particles, rume, likelihood_fn, model_link, index
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

                # Propagate particles and update their states
                propagated_particles = self.propagate_particles(
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
                propagated_particles = particles

            # Append model data (mean of particle states) for this time step
            model_data.append(
                np.mean(
                    [
                        np.array(particle.events_state)[:, index]
                        for particle in propagated_particles
                    ],
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
                    propagated_particles,
                )

                if np.any(np.isnan(new_weights)):
                    raise ValueError("NaN values found in computed weights.")
                particles = weights_resampling.resample_particles(
                    propagated_particles, new_weights
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
        print(f"\nSimulation completed in {total_runtime:.2f}s")
        print(f"\nParameters estimated: {parameters_estimated}")

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
