"""
Simulator and utility classes for running an ensemble Kalman filter for state and
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
    _FilterUpdateResult,
)


@dataclass(frozen=True)
class _EnsembleKalmanFilterContext:
    """
    Stores additional context and diagonistic information for the ensemble Kalman filter
    update.
    """


@dataclass(frozen=True)
class EnsembleKalmanFilterOutput(FilterOutput):
    """
    Output object for the ensemble Kalman filter.

    Parameters
    ----------
    simulator :
        The simulator used to produce the output.
    """

    simulator: "EnsembleKalmanFilterSimulator"
    """The simulator used to produce the output."""


@dataclass(frozen=True)
class EnsembleKalmanFilterSimulator(FilterSimulator[_EnsembleKalmanFilterContext]):
    """
    A PipelineSimulator for running an ensemble Kalman filter to estimate the state and
    parameters of a system.

    Parameters
    ----------
    config : PipelineConfig
        The RUME, number of realization, initial (prior) compartment values, and unknown
        parameters to estimate.
    observations : Observations
        The observations used to estimate the compartment and parameter values.
    """

    @override
    def _initialize_filter_context(self):
        return _EnsembleKalmanFilterContext()

    @override
    def _update(
        self,
        current_observation: NDArray,
        current_predictions: NDArray,
        prior_compartments: NDArray,
        prior_params: Mapping[NamePattern, NDArray],
        filter_context: _EnsembleKalmanFilterContext | None,
        rng: np.random.Generator,
    ) -> _FilterUpdateResult:
        """
        Run the update step of the ensemble Kalman filter (EnKF) for each node
        independently. The EnKF update can produce negative values so the
        compartments values are clipped to be positive after the update step.
        The Kalman gain matrix is decomposed into separate blocks in order to update the
        compartments, parameters, and predicted values separately. In particular this
        decomposition assumes the parameters are conditionally independent (conditioned
        on the compartment values) from the observed values.
        """
        posterior_value = current_predictions.copy().astype(np.float64)

        current_compartments = prior_compartments.copy()
        current_params = {k: prior_params[k].copy() for k in prior_params.keys()}

        # Hard code localization
        for i_node in range(self.rume.scope.nodes):
            observation_is_missing = (current_observation.size == 0) or np.ma.is_masked(
                current_observation["value"][i_node, ...]
            )

            if observation_is_missing:
                continue

            observation = current_observation["value"][i_node, ...]

            cov_const = 1 / (self.num_realizations - 1)
            simulated_observations = self.observations.likelihood.sample(
                current_predictions[:, i_node, ...], rng=rng
            )
            simulated_mean = simulated_observations.mean(axis=0, keepdims=True)
            simulated_dev = simulated_observations - simulated_mean
            residual_cov = cov_const * np.matmul(simulated_dev.T, simulated_dev)
            residual_cov_inverse = np.linalg.pinv(residual_cov)

            # Update compartments.
            compartments_mean = np.mean(current_compartments[:, i_node, :], axis=0)
            compartments_dev = (
                # Relies on numpy broadcasting rules, i.e. left padding shape.
                current_compartments[:, i_node, :] - compartments_mean
            )
            compartments_cross_cov = cov_const * np.matmul(
                compartments_dev.T, simulated_dev
            )
            kalman_gain_compartments = compartments_cross_cov @ residual_cov_inverse

            for i_ensemble in range(self.num_realizations):
                residual = (
                    observation - simulated_observations[i_ensemble, ...]
                ).reshape((-1, 1))

                compartments_correction = (
                    np.rint(kalman_gain_compartments @ residual)
                    .astype(np.int64)
                    .squeeze()
                )
                current_compartments[i_ensemble, i_node, :] += compartments_correction

            # Update estimated parameters.
            params_dev = {}
            for k in current_params.keys():
                params_mean = np.mean(current_params[k][:, i_node])
                params_dev[k] = current_params[k][:, i_node] - params_mean

            kalman_gain_params = {}
            for k in params_dev.keys():
                params_cross_cov = cov_const * np.matmul(params_dev[k], simulated_dev)
                kalman_gain_params[k] = params_cross_cov @ residual_cov_inverse

            for i_ensemble in range(self.num_realizations):
                residual = (
                    observation - simulated_observations[i_ensemble, ...]
                ).reshape((-1, 1))

                for name in kalman_gain_params.keys():
                    params_correction = (kalman_gain_params[name] @ residual).squeeze()
                    current_params[name][i_ensemble, i_node] += params_correction

            # Update estimation of observed data.
            prediction_mean = current_predictions[:, i_node, ...].mean(
                axis=0, keepdims=True
            )
            prediction_dev = current_predictions[:, i_node, ...] - prediction_mean

            prediction_cross_cov = cov_const * np.matmul(
                prediction_dev.T, simulated_dev
            )
            kalman_gain_prediction = prediction_cross_cov @ residual_cov_inverse

            for i_ensemble in range(self.num_realizations):
                residual = (
                    observation - simulated_observations[i_ensemble, ...]
                ).reshape((-1, 1))

                posterior_correction = (kalman_gain_prediction @ residual).squeeze()
                posterior_value[i_ensemble, i_node, ...] += posterior_correction

        current_compartments = np.clip(current_compartments, a_min=0, a_max=None)
        posterior_value = np.clip(posterior_value, a_min=0, a_max=None)
        return _FilterUpdateResult(
            current_compartments=current_compartments,
            current_params=current_params,
            posterior_values=posterior_value,
            filter_context=_EnsembleKalmanFilterContext(),
        )

    @override
    def run(self, rng):
        run_filter_result = self._run_filter(rng=rng)
        return EnsembleKalmanFilterOutput(
            simulator=self,
            final_compartments=run_filter_result.final_compartments,
            final_params=run_filter_result.final_params,
            compartments=run_filter_result.compartments,
            events=run_filter_result.events,
            initial=run_filter_result.initial,
            estimated_params=run_filter_result.estimated_params,
            posterior_values=run_filter_result.posterior_values,
            movement_data_mode=self.movement_data_mode,
        )
