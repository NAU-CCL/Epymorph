from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np

from epymorph.parameter_fitting.dynamics import GeometricBrownianMotion
from epymorph.parameter_fitting.filter.particle import Particle
from epymorph.parameter_fitting.output import ParticleFilterOutput
from epymorph.parameter_fitting.perturbation import Calvetti
from epymorph.parameter_fitting.utils import utils
from epymorph.parameter_fitting.utils.epymorph_simulation import EpymorphSimulation
from epymorph.parameter_fitting.utils.observations import ModelLink
from epymorph.parameter_fitting.utils.parameter_estimation import EstimateParameters
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
        params_space: Dict[str, EstimateParameters],
        rng: np.random.Generator,
    ) -> Tuple[List[Particle], List[np.ndarray]]:
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


class ForecastParameters:
    def __init__(
        self,
        initial_particles: List[Particle],
        rume: RUME,
        params_space: Dict[str, EstimateParameters],
        model_link: ModelLink,
        start_date: str,
        duration_weeks: int,
        time_interval: int = 7,
    ):
        self.initial_particles = initial_particles
        self.rume = rume
        self.params_space = params_space
        self.model_link = model_link
        self.start_date = start_date
        self.duration_weeks = duration_weeks
        self.time_interval = time_interval
        self.propagation = PropagateParticles()
        self.rng = np.random.default_rng()
        self.param_quantiles = {}
        self.param_values = {}

    from datetime import datetime, timedelta

    def calculate_end_date(self, start_date: str, duration_weeks: int):
        # Convert start_date string to a datetime object
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")

        # Calculate the total duration in days
        total_days = duration_weeks * 7

        # Calculate the end date by adding the total days to the start date
        end_date_obj = start_date_obj + timedelta(days=total_days)

        # Generate the dates list
        dates_list = []
        current_date = start_date_obj

        while current_date <= end_date_obj:
            dates_list.append(current_date.strftime("%Y-%m-%d"))
            current_date += timedelta(days=7)  # Adding 7 days to get the next date

        return end_date_obj.strftime("%Y-%m-%d"), dates_list

    def forecast_params(self):
        simulation = EpymorphSimulation(self.rume, self.start_date)

        end_date, dates_list = self.calculate_end_date(
            self.start_date, self.duration_weeks
        )

        # Prepare containers for storing results
        for key in self.params_space.keys():
            self.param_quantiles[key] = []
            self.param_values[key] = []

        for t in range(len(dates_list)):
            # Propagate particles and update their states
            propagated_particles, expected_observations = (
                self.propagation.propagate_particles(
                    self.initial_particles,
                    self.rume,
                    simulation,
                    dates_list[t],
                    self.time_interval,
                    self.model_link,
                    self.params_space,
                    self.rng,
                )
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
            model_data=np.array([]),
            particles=particles,  # type: ignore
        )

        return out
