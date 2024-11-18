"""
This module provides the ParticleInitializer class for initializing particles in the particle filter.
Each particle is initialized with a state and corresponding observations based on specified parameters.
"""

from typing import Any, Dict, List

import numpy as np

from epymorph.database import NamePattern
from epymorph.parameter_fitting.filters.particle import Particle
from epymorph.parameter_fitting.utils.parameter_estimation import EstimateParameters
from epymorph.simulator.data import evaluate_params, initialize_rume


class ParticleInitializer:
    """
    A class to initialize particles for the particle filter.

    Attributes:
        num_particles (int): Number of particles.
        num_population (int): Number of population.
        seed_size (int): Seed size.
        static_params (Dict[str, Any]): Static parameters.
        dynamic_params (Dict[str, Tuple[float, float]]): Dynamic parameters with their ranges.
        geo (Dict[str, Any]): Geographical information.
        nodes (int): Number of nodes in the geographical network.
    """

    def __init__(
        self,
        num_particles: int,
        rume: Any,
        dynamic_params: Dict[str, EstimateParameters],
    ) -> None:
        """
        Initializes the ParticleInitializer with the given parameters.

        Args:
            num_particles (int): Number of particles.
            rume (Dict[str, Any]): Dictionary containing model parameters including
            population size, seed size, static parameters, and geographical information.
            dynamic_params (Dict[str, Tuple[float, float]]): Dictionary containing
            dynamic parameters and their ranges.
        """
        self.num_particles = num_particles
        self.rume = rume
        self.dynamic_params = dynamic_params
        self.rng = np.random.default_rng()
        # self.geo = rume['geo']
        # self.nodes = self.geo.nodes

    def initialize_particles(self) -> List[Particle]:
        """
        Initializes particles with random values within the specified ranges for dynamic
        parameters.

        Returns:
            List[Tuple[np.ndarray, Dict[str, float]]]: A list of tuples where each tuple
            contains the initial state and the observations for a particle.
        """

        for _ in self.dynamic_params.keys():
            new_param = NamePattern(strata="*", module="*", id=_)

            self.rume.params[new_param] = 100
            self.rume.params

        rng = np.random.default_rng()
        data = evaluate_params(self.rume, {}, rng)
        initial_state = initialize_rume(self.rume, rng, data)
        initial_events_state = np.zeros_like(initial_state)

        particles = []

        for _ in range(self.num_particles):
            parameters = {
                _: self.dynamic_params[_].distribution.rvs(size=self.rume.dim.nodes)  # type: ignore
                for _ in self.dynamic_params.keys()
            }

            particle_state = initial_state

            particle = Particle(
                state=initial_state,
                parameters=parameters,
                events_state=initial_events_state,
            )

            particles.append(particle)

        return particles
