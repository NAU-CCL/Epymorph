"""
Simulator and utility classes for running a particle filter for state and
parameter estimation.
"""

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from epymorph.attribute import NamePattern
from epymorph.forecasting.filter import (
    FilterOutput,
    FilterSimulator,
    Observations,
    _FilterUpdateResult,
)
from epymorph.forecasting.pipeline import PipelineConfig


@dataclass(frozen=True)
class _ParticleFilterContext:
    """Additional diagnostics for the particle filter update."""

    effective_sample_size: np.ndarray
    """The effective sample size prior to resampling."""


@dataclass(frozen=True)
class ParticleFilterOutput(FilterOutput):
    """
    Output object for the particle filter which contains additional output and
    diagnostic information.

    Parameters
    ----------
    simulator :
        The simulator used to produce the output.
    posterior_values :
        The posterior estimate of the observed data. The first dimension of the array is
        the number of realizations, the second dimension is the number of observations.
        The remaining dimensions depend on the shape of the observed data. These values
        are constructed from resampling the observed data in the same way the
        compartment and parameter values are resampled.
    effective_sample_size :
        The effective sample size, defined as the the reciprocal of the sum of the
        squared weights, for each observation and each node.
    """

    simulator: "ParticleFilterSimulator"

    effective_sample_size: NDArray[np.float64]
    """
    The effective sample size for each observation and each node.
    """


@dataclass(frozen=True)
class ParticleFilterSimulator(FilterSimulator[_ParticleFilterContext]):
    """
    A PipelineSimulator for using a particle filter to estimate the state and parameters
    of a system based on observed data.

    Parameters
    ----------
    config :
        The RUME, number of realization, initial (prior) compartment values, and unknown
        parameters to estimate.
    observations :
        The observations used to estimate the compartment and parameter values.
    """

    config: PipelineConfig
    """The configuration for the simulation."""

    observations: Observations
    """The observations."""

    @staticmethod
    def _normalize_log_weights(log_weights: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Calculate the normalized particle weights from the un-normalized natural
        logarithm of the weights. This method uses multiple techniques to mitigate
        underflow issues caused by large population sizes and multi-node systems.

        Parameters
        ----------
        log_weights :
            The natural logarithm of the weights.

        Returns
        -------
        :
            The normalized (non-log) weights.
        """
        # Converting unnormalized log weights to normalized non-log weights is
        # essentially a soft max of the of the log weights.

        # Shifting the log weights, equivalent to multiplying the unnormalized non-log
        # weights by a constant, helps avoid underflow issues.
        shifted_log_weights = log_weights - np.max(log_weights, keepdims=True)
        underflow_lower_bound = -(10**2)
        clipped_log_weights = np.clip(
            shifted_log_weights, a_min=underflow_lower_bound, a_max=None
        )
        # Perform the softmax.
        weights = np.exp(clipped_log_weights)
        weights = weights / np.sum(weights, keepdims=True)
        return weights

    @staticmethod
    def _systematic_resampling(
        weights: NDArray, rng: np.random.Generator
    ) -> NDArray[np.int64]:
        """
        Performs systematic resampling as part of a particle filter update. It is used
        to take a weighted particle cloud and produce an equally-weighted particle cloud
        through resampling. Each particle in the resampled cloud is a duplicate of a
        particle in the orginal cloud. Not all particles are retained in the resampled
        cloud. An important property of systematic resampling is that a cloud of
        equally-weighted particles will produce the exact same cloud.


        Parameters
        ----------
        weights :
            An array of shape (R,) where R is the number of particles containing the
            particle weights.
        rng:
            The random number generator.

        Returns:
        :
            An array of shape (R,) where R is the number of particles containing the
            indices of the prior particles corresponding to the resampled particles.
        """
        num_realizations = len(weights)
        c = np.cumsum(weights)
        u = rng.uniform(0, 1)
        resampled_idx = []
        for i_realization in range(num_realizations):
            u_i = (u + i_realization) / num_realizations
            j = np.where(c >= u_i)[0][0]
            resampled_idx.append(j)
        return np.array(resampled_idx)

    def _initialize_filter_context(self):
        return _ParticleFilterContext(
            effective_sample_size=np.zeros(shape=(0, self.rume.scope.nodes))
        )

    @override
    def _update(
        self,
        current_observation: NDArray,
        current_predictions: NDArray,
        prior_compartments: NDArray,
        prior_params: Mapping[NamePattern, NDArray],
        filter_context: _ParticleFilterContext,
        rng: np.random.Generator,
    ) -> _FilterUpdateResult:
        """
        Update step of the particle filter. Each node is updated independently then the
        state of each particle is reassembled. The update step uses systematic
        resampling which is well-suited towards producing spatially continuous particles
        despite localization.
        """
        # posterior_value will be modified in-place.
        posterior_value = current_predictions.copy()

        current_compartments = prior_compartments.copy()
        current_params = {k: prior_params[k].copy() for k in prior_params.keys()}

        posterior_value = current_predictions.copy()
        prior_compartment_values = current_compartments.copy()

        effective_sample_size = np.zeros(shape=(1, self.rume.scope.nodes))

        # Hard code localization
        for i_node in range(self.rume.scope.nodes):
            observation_is_missing = (current_observation.size == 0) or np.ma.is_masked(
                current_observation["value"][i_node, ...]
            )

            if observation_is_missing:
                effective_sample_size[0, i_node] = 1.0
            else:
                log_likelihoods = self.observations.likelihood.compute_log(
                    current_observation["value"][i_node, ...],
                    current_predictions[:, i_node, ...],
                )
                weights = self._normalize_log_weights(log_likelihoods)

                resampled_idx = self._systematic_resampling(weights, rng)

                orig_idx = np.arange(0, self.num_realizations)
                current_compartments[orig_idx, i_node, ...] = prior_compartment_values[
                    resampled_idx, i_node, ...
                ]
                for name in current_params.keys():
                    current_params[name][orig_idx, i_node] = current_params[name][
                        resampled_idx, i_node
                    ]

                posterior_value[orig_idx, i_node, ...] = posterior_value[
                    resampled_idx, i_node, ...
                ]
                effective_sample_size[0, i_node] = 1 / np.sum(weights**2)
        return _FilterUpdateResult(
            current_compartments=current_compartments,
            current_params=current_params,
            posterior_values=posterior_value,
            filter_context=_ParticleFilterContext(
                effective_sample_size=np.concatenate(
                    [filter_context.effective_sample_size, effective_sample_size],
                    axis=0,
                )
            ),
        )

    @override
    def run(self, rng):
        run_filter_result = self._run_filter(rng=rng)
        return ParticleFilterOutput(
            simulator=self,
            final_compartments=run_filter_result.final_compartments,
            final_params=run_filter_result.final_params,
            compartments=run_filter_result.compartments,
            events=run_filter_result.events,
            initial=run_filter_result.initial,
            estimated_params=run_filter_result.estimated_params,
            posterior_values=run_filter_result.posterior_values,
            effective_sample_size=run_filter_result.filter_context.effective_sample_size,
        )
