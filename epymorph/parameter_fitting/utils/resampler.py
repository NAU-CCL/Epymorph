"""
This module defines the WeightsResampling class for computing and resampling particle weights in particle filtering.

The WeightsResampling class handles the calculation of weights based on the likelihood of observations, resampling
particles based on these weights, and updating weights as needed.
"""

import numpy as np

rng = np.random.default_rng()


class WeightsResampling:
    """
    A class for computing and resampling weights in particle filtering.

    Attributes:
        N (int): Number of particles.
        rume (dict): A dictionary containing simulation parameters including
        'static_params' and 'ipm'.
        static_params (dict): Static parameters from the rume dictionary.
        likelihood_fn (object): An object responsible for computing the
        likelihood of observations.
        model_link (str): A string representing the model link used for indexing.

    Methods:
        get_index() -> int:
            Retrieves the index of the model link in the IPM or compartment names.

        compute_weights(current_obs_data: np.ndarray, particles: list) -> np.ndarray:
            Computes the weights for each particle based on the likelihood
            of the current observation.

        resample_particles(particles: list, weights: np.ndarray) -> list:
            Resamples particles based on the computed weights.

        update_weights(weights: np.ndarray, new_weights: np.ndarray) -> list:
            Updates the weights by multiplying current weights with new weights.
    """

    def __init__(
        self,
        N: int,
        rume: dict,
        likelihood_fn,
        model_link: str,
        index: int,
    ):
        """
        Initializes the WeightsResampling class with provided parameters.

        Args:
            N (int): Number of particles.
            rume (dict): A dictionary of parameters required for simulation.
            likelihood_fn (object): An object for computing the likelihood of
            observations.
            model_link (str): A string representing the model link used for indexing.
        """
        self.N = N
        self.rume = rume
        self.likelihood_fn = likelihood_fn
        self.model_link = model_link
        self.index = index

    def get_index(self) -> int:
        """
        Retrieves the index of the model link in the IPM or compartment names.

        Returns:
            int: The index of the model link.
        """
        if "->" in self.model_link or "→" in self.model_link:
            index = self.rume["ipm"].event_names.index(self.model_link)
        else:
            index = self.rume["ipm"].compartment_names.index(self.model_link)

        return index

    def compute_weights(self, current_obs_data: np.ndarray, states: list) -> np.ndarray:
        """
        Computes the weights for each particle based on the likelihood of the
        current observation.

        Args:
            current_obs_data (np.ndarray): The current observation data.
            states (list): A list of states, where each state is a tuple of
            state and observations.

        Returns:
            np.ndarray: An array of computed weights normalized to sum to 1.
        """
        log_weights = np.zeros(self.N)

        for i in range(self.N):
            expected = states[i][0][:, self.index]
            try:
                log_weights[i] = np.sum(
                    np.log(
                        self.likelihood_fn.compute(current_obs_data, expected + 0.0001)
                    )
                )
            except (ValueError, FloatingPointError):
                log_weights[i] = -np.inf
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

    def resample_particles(self, particles: list, weights: np.ndarray) -> list:
        """
        Resamples particles based on the computed weights.

        Args:
            particles (list): A list of particles, where each particle is a tuple of
            state and observations.
            weights (np.ndarray): An array of weights corresponding to each particle.

        Returns:
            list: A list of resampled particles.
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

        Args:
            weights (np.ndarray): The current weights of the particles.
            new_weights (np.ndarray): The new weights to be multiplied with the
            current weights.

        Returns:
            list: A list of updated weights.
        """
        updated_weights = [
            new_weight * weight for weight, new_weight in zip(weights, new_weights)
        ]
        return updated_weights


class ResamplingByNode(WeightsResampling):
    def compute_weights(self, current_obs_data: np.ndarray, states: list) -> np.ndarray:
        """
        Computes the weights for each particle based on the likelihood of the
        current observation.

        Args:
            current_obs_data (np.ndarray): The current observation data.
            states (list): A list of states, where each state is a tuple of
            state and observations.

        Returns:
            np.ndarray: An array of computed weights normalized to sum to 1.
        """

        n_nodes = states[0][0].shape[0]
        log_weights = np.zeros(shape=(n_nodes, self.N))

        for i_node in range(n_nodes):
            for i_particle in range(self.N):
                expected = states[i_particle][0][i_node, self.index]
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

    def resample_particles(self, particles: list, weights: np.ndarray) -> list:
        """
        Resamples particles based on the computed weights.

        Args:
            particles (list): A list of particles, where each particle is a tuple of
            state and observations.
            weights (np.ndarray): An array of weights corresponding to each particle.

        Returns:
            list: A list of resampled particles.
        """
        n_nodes = weights.shape[0]
        # for i_node in range(n_nodes):

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
            new_x = np.copy(particles[i_particle][0])
            new_beta = np.copy(particles[i_particle][1]["beta"])

            for i_node in range(n_nodes):
                new_x[i_node, :] = particles[resampled_indices[i_node, i_particle]][0][
                    i_node, :
                ]
                new_beta[i_node] = particles[resampled_indices[i_node, i_particle]][1][
                    "beta"
                ][i_node]
            resampled_particles.append((new_x, {"beta": new_beta}))

        # [resampled_particles.append(particles[idx]) for idx in resampled_indices[0, :]]

        return resampled_particles
