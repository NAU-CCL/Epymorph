import numpy as np

from epymorph.parameter_fitting.filter.particle import Particle
from epymorph.parameter_fitting.utils.observations import ModelLink


class WeightsResampling:
    """
    A class for computing and resampling weights in particle filtering.

    Attributes
    ----------
    N : int
        Number of particles.
    static_params : dict
        Static parameters from the rume dictionary.
    likelihood_fn : object
        An object responsible for computing the
        likelihood of observations.
    model_link : ModelLink
        Represents the model link used for indexing.
    """

    def __init__(
        self,
        N: int,
        likelihood_fn,
        model_link: ModelLink,
    ) -> None:
        """
        Initializes the WeightsResampling class with provided parameters.

        Parameters
        ----------
        N : int
            Number of particles.
        likelihood_fn : object
            An object for computing the likelihood of
            observations.
        model_link : ModelLink
            A string representing the model link used for
            indexing.
        """
        self.N = N
        self.likelihood_fn = likelihood_fn
        self.model_link = model_link

    def compute_weights(
        self, current_obs_data: np.ndarray, expected_observations: list
    ) -> np.ndarray:
        """
        Computes the weights for each particle based on the likelihood of the
        current observation.

        Parameters
        ----------
        current_obs_data : np.ndarray
            The current observation data.
        expected_observations : list
            A list of expected observations corresponding
            to each particle.

        Returns
        -------
        np.ndarray
            An array of computed weights normalized to sum to 1.
        """
        log_weights = np.zeros(self.N)

        for i in range(self.N):
            expected = expected_observations[i]
            try:
                log_weights[i] = np.sum(
                    np.log(
                        self.likelihood_fn.compute(current_obs_data, expected + 0.0001)
                    )
                )
            except (ValueError, FloatingPointError):
                log_weights[i] = -np.inf

        # Compute the softmax of log_weights, which yields the normalized weights.
        try:
            shifted_log_weights = log_weights - np.max(log_weights, keepdims=True)
        except FloatingPointError:
            shifted_log_weights = np.zeros(shape=log_weights.shape)
        underflow_lower_bound = -(10**2)
        clipped_log_weights = np.clip(
            shifted_log_weights, a_min=underflow_lower_bound, a_max=None
        )
        weights = np.exp(clipped_log_weights)
        weights = weights / np.sum(weights, keepdims=True)

        return np.squeeze(weights)

    def resample_particles(
        self, particles: list, weights: np.ndarray, rng: np.random.Generator
    ) -> list:
        """
        Resamples particles based on the computed weights.

        Parameters
        ----------
        particles : list
            A list of particles, where each particle is a tuple of
            state and observations.
        weights : np.ndarray
            An array of weights corresponding to each particle.

        Returns
        -------
        list
            A list of resampled particles.
        """
        resampled_particles = []
        resampled_indices = rng.choice(
            np.arange(self.N), size=self.N, replace=True, p=weights
        )
        [resampled_particles.append(particles[idx]) for idx in resampled_indices]
        return resampled_particles

    def update_weights(self, weights: np.ndarray, new_weights: np.ndarray) -> list:
        """
        Updates the weights by multiplying current weights with new weights.

        Parameters
        ----------
        weights : np.ndarray
            The current weights of the particles.
        new_weights : np.ndarray
            The new weights to be multiplied with the
            current weights.

        Returns
        -------
        list
            A list of updated weights.
        """
        updated_weights = [
            new_weight * weight for weight, new_weight in zip(weights, new_weights)
        ]
        return updated_weights


class ResamplingByNode(WeightsResampling):
    def compute_weights(
        self, current_obs_data: np.ndarray, expected_observations: list
    ) -> np.ndarray:
        n_nodes = expected_observations[0].shape[0]
        log_weights = np.zeros(shape=(n_nodes, self.N))

        for i_node in range(n_nodes):
            for i_particle in range(self.N):
                expected = expected_observations[i_particle][i_node]
                try:
                    log_weights[i_node, i_particle] = np.log(
                        self.likelihood_fn.compute(
                            current_obs_data[i_node], expected + 0.0001
                        )
                    )
                except (ValueError, FloatingPointError):
                    log_weights[i_node, i_particle] = -np.inf

        shifted_log_weights = log_weights - np.max(log_weights, axis=1, keepdims=True)
        underflow_lower_bound = -(10**2)
        clipped_log_weights = np.clip(
            shifted_log_weights, a_min=underflow_lower_bound, a_max=None
        )
        weights = np.exp(clipped_log_weights)
        weights = weights / np.sum(weights, axis=1, keepdims=True)

        return weights

    def resample_particles(
        self, particles: list[Particle], weights: np.ndarray, rng: np.random.Generator
    ) -> list:
        n_nodes = weights.shape[0]

        resampled_indices = np.zeros(shape=(n_nodes, self.N), dtype=np.int_)
        for i_node in range(n_nodes):
            resampled_indices[i_node, :] = rng.choice(
                np.arange(self.N, dtype=np.int_),
                size=self.N,
                replace=True,
                p=weights[i_node, :],
            )

        resampled_particles = []
        for i_particle in range(self.N):
            new_state = np.zeros_like(particles[i_particle].state)
            new_parameters = {}

            for i_node in range(n_nodes):
                new_state[i_node, :] = particles[
                    resampled_indices[i_node, i_particle]
                ].state[i_node, :]

            for key, value in particles[i_particle].parameters.items():
                new_value = np.zeros_like(value)
                for i_node in range(n_nodes):
                    idx = resampled_indices[i_node, i_particle]
                    new_value[i_node] = particles[idx].parameters[key][i_node]
                new_parameters[key] = new_value

            resampled_particles.append(
                Particle(
                    new_state,
                    new_parameters,
                )
            )

        return resampled_particles
