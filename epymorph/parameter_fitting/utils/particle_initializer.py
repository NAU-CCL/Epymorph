"""
This module provides the ParticleInitializer class for
initializing particles in the particle filter.
Each particle is initialized with a state and corresponding
observations based on specified parameters.
"""

from typing import Dict, List

import numpy as np

from epymorph.parameter_fitting.filter.particle import Particle
from epymorph.parameter_fitting.utils.parameter_estimation import EstimateParameters
from epymorph.rume import RUME


class ParticleInitializer:
    """
    A class to initialize particles for the particle filter.

    Attributes
    ----------
    num_particles : int
        Number of particles.
    num_population : int
        Number of population.
    seed_size : int
        Seed size.
    static_params : Dict[str, Any])
        Static parameters.
    dynamic_params : Dict[str, EstimateParameters]
        Dynamic parameters with their ranges.
    geo : Dict[str, Any]
        Geographical information.
    nodes : int
        Number of nodes in the geographical network.
    """

    def __init__(
        self,
        num_particles: int,
        rume: RUME,
        dynamic_params: Dict[str, EstimateParameters],
    ) -> None:
        """
        Initializes the ParticleInitializer with the given parameters.

        Parameters
        ----------
        num_particles : int
            Number of particles.
        rume : Rume
            Model parameters including population size, seed size, static parameters,
            and geographical information.
        dynamic_params : Dict[str, EstimateParameters]
            Dynamic parameters and their ranges.
        """
        self.num_particles = num_particles
        self.rume = rume
        self.dynamic_params = dynamic_params
        self.rng = np.random.default_rng()

    def initialize_particles(self, rng) -> List[Particle]:
        """
        Initializes particles with random values within the specified ranges for dynamic
        parameters.

        Returns
        -------
        List[Particle]
            The initialized particles.
        """

        data = self.rume.evaluate_params(rng)
        initial_state = self.rume.initialize(data, rng)

        particles = []

        for _ in range(self.num_particles):
            parameters = {
                _: self.dynamic_params[_].distribution.rvs(
                    size=self.rume.scope.nodes, random_state=rng
                )
                for _ in self.dynamic_params.keys()
            }

            # Create a Particle instance with the initial state and parameters
            particle = Particle(
                state=initial_state,
                parameters=parameters,
            )

            particles.append(particle)

        return particles
