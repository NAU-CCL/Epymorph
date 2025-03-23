import dataclasses
from typing import Callable, Generic, TypeVar

import numpy as np

from epymorph import initializer
from epymorph.parameter_fitting.dynamics import GeometricBrownianMotion
from epymorph.parameter_fitting.utils import utils
from epymorph.parameter_fitting.utils.params_perturb import Perturb
from epymorph.rume import RUME
from epymorph.simulator.basic.basic_simulator import BasicSimulator

RUMEType = TypeVar("RUMEType", bound=RUME)


class ForecastSimulator(Generic[RUMEType]):
    rume: RUMEType
    """The RUME we will use for the simulation."""

    def __init__(self, rume: RUMEType):
        self.rume = rume

    def run(
        self,
        particles,
        params_space,
        rng_factory: Callable[[], np.random.Generator] | None = None,
    ):
        rume = self.rume

        duration = rume.time_frame.days

        rng = (rng_factory or np.random.default_rng)()

        # List to store the observation results
        observation_results = []

        # Dictionaries to store quantiles and mean values of each parameter over time
        param_quantiles = {key: [] for key in params_space.keys()}
        param_means = {key: [] for key in params_space.keys()}

        all_particle_params_per_day = []

        # Initialize particle parameters for the first time step
        particle_params = [particle.parameters for particle in particles]

        # Propagate parameters for each time step
        for t in range(duration):
            # Propagate parameters based on the current state
            propagated_params = self.propagate_params(
                particle_params, params_space, rng
            )

            # Dictionary to temporarily store parameter values for quantiles and means
            temp_key_values = {key: [] for key in param_quantiles.keys()}

            # Collect values for each parameter across all particles
            for param_set in propagated_params:
                for key in temp_key_values:
                    if key in param_set:
                        temp_key_values[key].append(param_set[key])

            # Calculate and store the quantiles and means for each parameter
            for key, values in temp_key_values.items():
                if values:
                    param_quantiles[key].append(utils.quantiles(np.array(values)))
                    param_means[key].append(np.mean(values))
                    all_particle_params_per_day.append(
                        {key: np.concatenate(values).tolist()}
                    )
            # Update particle parameters for the next time step
            particle_params = propagated_params

        # After the final time step, create a dictionary of the final mean values of the parameters
        final_param_values = {key: np.array(param_means[key]) for key in param_means}

        # Propagate each particle using the final mean values
        for particle in particles:
            # Apply the model to each particle and collect the results
            result = self.propagate_particle(rume, particle, final_param_values, rng)
            observation_results.append(result)

        # Return all collected data: observations, final parameters, and quantiles/means of parameters
        return (
            observation_results,
            final_param_values,
            param_quantiles,
            param_means,
            np.array(all_particle_params_per_day),
        )

    def propagate_params(self, parameters, params_space, rng):
        # Update the parameters using their dynamics
        params_perturb = Perturb(1)
        propagated_parameters = []
        for parameter in parameters:
            new_parameters = {}
            for param, val in parameter.items():
                dynamics = params_space[param].dynamics
                if isinstance(dynamics, GeometricBrownianMotion):
                    new_parameters[param] = params_perturb.gbm(
                        val, dynamics.volatility, rng
                    )
                else:
                    new_parameters[param] = val

            propagated_parameters.append(new_parameters)

        return propagated_parameters

    def propagate_particle(self, rume, particle, parameters, rng):
        # Create a copy of the RUME model with updated parameters and time frame
        rume_propagate = dataclasses.replace(
            rume,
            # time_frame=TimeFrame.of(date, duration),  # Set simulation duration
            strata=[
                dataclasses.replace(
                    g, init=initializer.Explicit(initials=particle.state)
                )  # Initialize with state values
                for g in rume.strata  # For each stratum, set the initial state
            ],
        )

        # Initialize the simulation using the BasicSimulator from the Epymorph library
        sim = BasicSimulator(rume_propagate)

        # Run the simulation and collect the output based on observations
        # (dynamic params)
        output = sim.run(parameters, rng_factory=(lambda: rng))

        return output
