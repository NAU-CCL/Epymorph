from datetime import timedelta
from typing import Dict, List, Tuple

import numpy as np

from epymorph.parameter_fitting.dynamics import GeometricBrownianMotion
from epymorph.parameter_fitting.filter.particle import Particle
from epymorph.parameter_fitting.output import ParticleFilterOutput
from epymorph.parameter_fitting.perturbation import Calvetti
from epymorph.parameter_fitting.utils import utils
from epymorph.parameter_fitting.utils.epymorph_simulation import EpymorphSimulation
from epymorph.parameter_fitting.utils.observations import ModelLink
from epymorph.parameter_fitting.utils.parameter_estimation import ForecastParameters
from epymorph.parameter_fitting.utils.params_perturb import Perturb
from epymorph.rume import RUME


class PropagateParticles:
    def __init__(self):
        pass

    def propagate_particles(
        self,
        particles: List[Particle],
        rume: RUME,
        simulation: EpymorphSimulation,
        date: str,
        duration: int,
        model_link: ModelLink,
        params_space: Dict[str, ForecastParameters],
        rng: np.random.Generator,
    ) -> Tuple[List[Particle], List[np.ndarray]]:
        propagated_particles = []
        expected_observations = []

        # Initialize perturbation handler
        params_perturb = Perturb(duration)
        # print("date = ", date)
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


class ForecastSimulation:
    def __init__(
        self,
        rume: RUME,
        params_space: Dict[str, ForecastParameters],
        model_link: ModelLink,
        duration: int,
        initial_particles: List[Particle],
        # num_particles: Optional[int] = None,
    ):
        self.initial_particles = initial_particles
        # self.num_particles = num_particles
        self.rume = rume
        self.params_space = params_space
        self.model_link = model_link
        self.duration = duration
        self.propagation = PropagateParticles()
        self.rng = np.random.default_rng()
        self.param_quantiles = {}
        self.param_values = {}

        # Check if initial_particles is None, and generate if num is provided
        # if self.initial_particles is None:
        #     if self.num_particles is None:
        #         raise ValueError(
        #             "Either 'initial_particles' or 'num' must be provided."
        #         )

        #     initializer = ParticleInitializer(
        #         self.num_particles, rume, params_space
        #     )  # Generate particles if num is provided

        #     rng = np.random.default_rng(seed=1)

        #     self.initial_particles = initializer.initialize_particles(rng)

    def run(self):
        start_date = self.rume.time_frame.end_date
        simulation = EpymorphSimulation(self.rume, start_date.strftime("%Y-%m-%d"))

        # Prepare containers for storing results
        for key in self.params_space.keys():
            self.param_quantiles[key] = []
            self.param_values[key] = []

        model_data = []
        model_data_quantiles = []

        particles = self.initial_particles

        for t in range(self.duration):
            start_date = start_date + timedelta(days=7)
            # Propagate particles and update their states
            propagated_particles, expected_observations = (
                self.propagation.propagate_particles(
                    particles,
                    self.rume,
                    simulation,
                    start_date.strftime("%Y-%m-%d"),
                    7,
                    self.model_link,
                    self.params_space,
                    self.rng,
                )
            )

            # Append model data (mean of particle states) for this time step
            model_data.append(
                np.mean(
                    [obs for obs in expected_observations],
                    axis=0,
                ).astype(int)  # Ensure the final mean is also an integer
            )

            model_data_quantiles.append(
                utils.quantiles(np.array(expected_observations))
            )

            particles = propagated_particles.copy()

            for param in particles[0].parameters.keys():
                perturbation = self.params_space[param].perturbation
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
                    rvs = self.rng.multivariate_normal(
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

        # Prepare the output object
        out = ParticleFilterOutput(
            len(self.initial_particles),
            parameters_estimated,
            str(self.rume.time_frame.duration_days) + " days",
            self.param_quantiles,
            self.param_values,
            true_data=np.array([]),
            model_data=np.array(model_data),
            particles=particles,  # type: ignore
        )

        return out, model_data_quantiles
