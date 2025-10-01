import dataclasses
import datetime
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from epymorph.adrio.adrio import ADRIO
from epymorph.attribute import NamePattern
from epymorph.forecasting.dynamic_params import ParamFunctionDynamics, Prior
from epymorph.initializer import Explicit, Initializer
from epymorph.rume import RUME
from epymorph.simulation import Context
from epymorph.simulator.basic.basic_simulator import BasicSimulator
from epymorph.time import Dim, TimeFrame
from epymorph.tools.data import munge


class MultiRealizationSimulator: ...


class MultiRealizationConfig: ...


class FromRUME(MultiRealizationConfig):
    def __init__(
        self,
        rume,
        num_realizations,
        initial_values=None,
        param_values=None,
        unknown_params=None,
    ):
        self.rume = rume
        self.num_realizations = num_realizations
        self.initial_values = initial_values
        self.param_values = param_values
        self.unknown_params = unknown_params

    def configure(self):
        return dict(
            rume=self.rume,
            num_realizations=self.num_realizations,
            initial_values=self.initial_values,
            param_values=self.param_values,
            unknown_params=self.unknown_params,
        )


class FromOutput(MultiRealizationConfig):
    def __init__(self, output, override_dynamics={}):
        override_dynamics = {NamePattern.of(k): v for k, v in override_dynamics.items()}
        new_time_frame = TimeFrame.of(
            start_date=output.rume.time_frame.start_date
            + datetime.timedelta(days=output.duration),
            duration_days=output.rume.time_frame.duration_days - output.duration,
        )
        self.rume = replace(output.rume, time_frame=new_time_frame)
        self.num_realizations = output.num_realizations
        self.initial_values = output.final_compartment_values
        self.param_values = None
        self.unknown_params = {
            k: UnknownParam(prior=output.final_param_values[k], dynamics=v.dynamics)
            for k, v in output.unknown_params.items()
        }
        for k, v in override_dynamics.items():
            name = NamePattern.of(k)
            self.unknown_params[name] = UnknownParam(
                prior=self.unknown_params[name].prior, dynamics=v
            )

    def configure(self):
        return dict(
            rume=self.rume,
            num_realizations=self.num_realizations,
            initial_values=self.initial_values,
            param_values=self.param_values,
            unknown_params=self.unknown_params,
        )


class Observations:
    def __init__(self, source, model_link, likelihood, strict_labels=False):
        self.source = source
        self.model_link = model_link
        self.likelihood = likelihood
        self.strict_labels = strict_labels

    def _get_observations_dataframe(self, context):
        if isinstance(self.source, pd.DataFrame):
            return self.source
        elif isinstance(self.source, ADRIO):
            inspect = self.source.with_context_internal(context).inspect()
            if isinstance(inspect.source, pd.DataFrame):
                return inspect.source


# class ParticleFilterSimulation(MultiRealizationSimulation):
#     @staticmethod
#     def _run_particle_filter():
#         ...


@dataclass(frozen=True)
class UnknownParam:
    prior: NDArray | Prior
    dynamics: ParamFunctionDynamics


class ParticleFilterSimulator(MultiRealizationSimulator):
    rume: RUME
    num_realizations: int
    unknown_params: Mapping[NamePattern, UnknownParam]
    initial_compartment_values: NDArray | Sequence[NDArray | Initializer] | None

    def __init__(
        self, config, duration, observations, local_blocks=None, observation_mask=None
    ):
        self.config = config
        self.duration = duration
        self.observations = observations
        self.local_blocks = local_blocks
        self.observation_mask = observation_mask

    def run(self, rng):
        return self._run_particle_filter(
            **self.config.configure(),
            observations=self.observations,
            duration=self.duration,
            local_blocks=self.local_blocks,
            observation_mask=self.observation_mask,
            rng=rng,
        )

    def _run_particle_filter(
        self,
        rume,
        num_realizations,
        initial_values,
        param_values,
        unknown_params,
        duration,
        observations,
        local_blocks,
        observation_mask,
        rng,
    ):
        """

        Parameters
        ----------
        rume :
            The RUME.
        num_realizations :
            The number of realizations.
        initial_values :
            The initial compartment values across each realization. The first dimension
            is the realization dimension.
        param_values :
            The parameter values. Overrides values which appear in unknown_params.
        unknown_params :
            The estimated parameter values.
        duration :
            The duration of the forecast.
        observations:
            The observations.
        rng :
            The random number generator.
        Returns
        -------
        :
            The forecast output.
        """
        assert param_values is None
        unknown_params = {NamePattern.of(k): v for k, v in unknown_params.items()}
        original_rume = rume
        sim_time_frame = TimeFrame.of(
            original_rume.time_frame.start_date, duration_days=duration
        )
        rume = dataclasses.replace(
            original_rume,
            time_frame=sim_time_frame,
        )

        # TODO # rume = self._cache_adrios(...)
        context = Context.of(scope=rume.scope, time_frame=rume.time_frame, rng=rng)
        new_params = {}
        for name, param in rume.params.items():
            if isinstance(param, ADRIO):
                new_params[name] = param.with_context_internal(context).evaluate()
            else:
                new_params[name] = param

        rume = dataclasses.replace(
            rume,
            params=new_params,
        )

        # Dimension of the system.
        days = rume.time_frame.days
        taus = rume.num_tau_steps
        R = num_realizations
        T = days
        S = days * taus
        N = rume.scope.nodes
        C = rume.ipm.num_compartments
        E = rume.ipm.num_events

        # Allocate return values.
        initial = np.zeros(shape=(R, N, C), dtype=np.int64)
        compartments = np.zeros(shape=(R, S, N, C), dtype=np.int64)
        events = np.zeros(shape=(R, S, N, E), dtype=np.int64)
        estimated_params = {
            k: np.zeros(shape=(R, T, N), dtype=np.float64)
            for k in unknown_params.keys()
        }

        # Initialize the initial compartment and parameter values.
        current_compartment_values, current_param_values = (
            self._initialize_augmented_state_space(
                rume, unknown_params, num_realizations, initial_values, rng
            )
        )
        initial = current_compartment_values

        # Determine the number of observations and their associated time frames.
        time_frames, labels = self._observation_time_frames(
            rume, observations.model_link.time
        )
        resampling_indices = np.zeros(shape=(R, len(time_frames), N), dtype=np.int32)
        resampling_days = np.zeros(shape=(len(time_frames),), dtype=np.int32)
        resampling_steps = np.zeros(shape=(len(time_frames),), dtype=np.int32)

        # Determine the local blocks
        if local_blocks is not None:
            block_ids = np.unique(local_blocks.grouping.map(rume.scope.node_ids))
        else:
            block_ids = np.array(["all"])

        block_mask = []
        if local_blocks is not None:
            for i_block in range(len(block_ids)):
                block_mask.append(
                    local_blocks.grouping.map(rume.scope.node_ids) == block_ids[i_block]
                )
        else:
            block_mask = np.ones(rume.scope.nodes, dtype=bool)
        block_mask = np.array(block_mask)

        context = Context.of(
            scope=rume.scope,
            time_frame=rume.time_frame,
            ipm=rume.ipm,
            rng=np.random.default_rng(0),
        )
        observations_df = observations._get_observations_dataframe(context)

        observed_values = []
        predicted_values = []
        posterior_values = []

        day_idx = 0
        step_idx = 0
        for i_observation in range(len(time_frames)):
            time_frame = time_frames[i_observation]

            day_right = day_idx + time_frame.days
            step_right = step_idx + time_frame.days * rume.num_tau_steps

            # if not observations.strict_labels:
            #     print("Got here!")

            # observation_df = observations.source.loc[
            #     observations.source["time"] == labels[i_observation]
            # ].drop("time", axis=1)

            # TODO # self._get_current_observation(...)
            time_key = observations_df.keys()[0]
            geo_key = observations_df.keys()[1]

            if not observations.strict_labels and np.issubdtype(
                observations_df.dtypes[time_key], np.datetime64().dtype
            ):
                start = np.datetime64(time_frame.start_date.isoformat())
                end = np.datetime64(time_frame.end_date.isoformat())
                observation_df = observations_df.loc[
                    (observations_df[time_key] >= start)
                    & (observations_df[time_key] <= end)
                ].drop(time_key, axis=1)
            else:
                observation_df = observations_df.loc[
                    observations_df[time_key] == labels[i_observation]
                ].drop(time_key, axis=1)
            observation_df = observation_df.sort_values(geo_key)

            observed_values.append(observation_df.drop([geo_key], axis=1).to_numpy())
            # observed_values.append(observation_df)

            print(  # noqa: T201
                (
                    f"Observation: {i_observation}, "
                    f"Label: {labels[i_observation]}, "
                    f"Time Frame: {time_frame}"
                )
            )

            block_weights = np.zeros(
                (len(block_ids), num_realizations), dtype=np.float64
            )
            # for i_block in range(len(block_ids)):

            current_predictions = []
            weights = np.zeros(num_realizations, dtype=np.float64)
            log_weights = np.zeros(num_realizations, dtype=np.float64)
            log_weights_by_node = np.zeros(
                (num_realizations, N), dtype=np.float64
            )  # TODO
            for j_realization in range(num_realizations):
                rume_temp, data = self._create_sim_rume(
                    rume,
                    j_realization,
                    unknown_params,
                    current_param_values,
                    current_compartment_values,
                    time_frame,
                    rng,
                )

                sim = BasicSimulator(rume_temp)
                out_temp = sim.run(rng_factory=lambda: rng)

                # TODO # prediction = self._calculate_prediction(...)
                temp_time_agg = out_temp.rume.time_frame.select.all().agg()
                new_time_agg = dataclasses.replace(
                    temp_time_agg,
                    aggregation=observations.model_link.time.aggregation,
                )

                predicted_df = munge(
                    out_temp,
                    geo=observations.model_link.geo,
                    time=new_time_agg,
                    quantity=observations.model_link.quantity,
                )

                prediction = predicted_df.drop(["geo", "time"], axis=1)
                current_predictions.append(prediction.to_numpy())
                # current_predictions.append(predicted_df)

                # TODO # block_log_weights[j_realization, :] = self._compute_block_log_weights(...)
                node_log_likelihoods = observations.likelihood.compute_log(
                    observation_df.drop([geo_key], axis=1).to_numpy(),
                    prediction.to_numpy(),
                )
                log_weights[j_realization] = np.sum(node_log_likelihoods)
                log_weights_by_node[j_realization, ...] = node_log_likelihoods[
                    :, 0
                ]  # TODO
                if observation_mask is not None:
                    for i_block in range(len(block_ids)):
                        block_weights[i_block, j_realization] = np.sum(
                            node_log_likelihoods[observation_mask[i_block, ...], ...]
                        )
                else:
                    block_weights[:, j_realization] = np.sum(node_log_likelihoods)

                compartments[j_realization, step_idx:step_right, ...] = (
                    out_temp.compartments
                )
                events[j_realization, step_idx:step_right, ...] = out_temp.events
                current_compartment_values[j_realization, ...] = out_temp.compartments[
                    -1, ...
                ]

                for k in unknown_params.keys():
                    estimated_params[k][j_realization, day_idx:day_right, ...] = (
                        data.get_raw(k)
                    )
                    current_param_values[k][j_realization, ...] = data.get_raw(k)[
                        -1, ...
                    ]

            predicted_values.append(np.array(current_predictions))

            posterior_value = predicted_values[-1].copy()

            # predicted_values.append(current_predictions)

            # for i_block in range(len(block_ids)):
            #     weights = self._normalize_log_weights(block_weights[i_block, ...])

            #     # Needed because of numpy advanced indexing quirks.
            #     orig_idx = np.arange(0, num_realizations)[:, np.newaxis]

            #     resampled_idx = self._systematic_resampling(weights, rng)[:, np.newaxis]

            #     current_compartment_values[orig_idx, block_mask[i_block, ...], ...] = (
            #         current_compartment_values[
            #             resampled_idx, block_mask[i_block, ...], ...
            #         ]
            #     )
            #     for k in current_param_values.keys():
            #         current_param_values[k][orig_idx, block_mask[i_block, ...], ...] = (
            #             current_param_values[k][
            #                 resampled_idx, block_mask[i_block, ...], ...
            #             ]
            #         )

            #     resampling_indices[:, i_observation] = resampled_idx[:, 0]

            # Hard code no localization
            for i_node in range(N):
                weights = self._normalize_log_weights(log_weights_by_node[..., i_node])
                orig_idx = np.arange(0, num_realizations)[:, np.newaxis]
                resampled_idx = self._systematic_resampling(weights, rng)[:, np.newaxis]
                # resampled_idx = self._multinomial_resampling(weights, rng)[
                #     :, np.newaxis
                # ]

                current_compartment_values[orig_idx, i_node, ...] = (
                    current_compartment_values[resampled_idx, i_node, ...]
                )
                for k in current_param_values.keys():
                    current_param_values[k][orig_idx, i_node] = current_param_values[k][
                        resampled_idx, i_node
                    ]

                posterior_value[orig_idx, i_node, ...] = posterior_value[
                    resampled_idx, i_node, ...
                ]

                resampling_indices[:, i_observation, i_node] = resampled_idx[:, 0]

            posterior_values.append(posterior_value)
            # weights = self._normalize_log_weights(log_weights)

            # # resampled_idx = self._multinomial_resampling(weights, rng)
            # resampled_idx = self._systematic_resampling(weights, rng)

            # current_compartment_values = current_compartment_values[resampled_idx, ...]
            # for k in current_param_values.keys():
            #     current_param_values[k] = current_param_values[k][resampled_idx, ...]

            resampling_steps[i_observation] = step_right
            resampling_days[i_observation] = day_right

            day_idx = day_right
            step_idx = step_right

        # for k in estimated_params.keys():
        #     estimated_params[k][:, -1, ...] = estimated_params[k][
        #         resampling_indices[:, -1], -1, ...
        #     ]

        return SimpleNamespace(
            rume=original_rume,
            sim_rume=rume,
            duration=duration,
            initial=initial,
            compartments=compartments,
            events=events,
            num_realizations=num_realizations,
            estimated_params=estimated_params,
            unknown_params=unknown_params,
            final_compartment_values=current_compartment_values,
            final_param_values=current_param_values,
            predicted_values=np.array(predicted_values),
            posterior_values=np.array(posterior_values),
            observed_values=np.array(observed_values),
            resampling_indices=resampling_indices,
            resampling_days=resampling_days,
            resampling_steps=resampling_steps,
            # predicted_values=predicted_values,
            # observed_values=observed_values,
        )

    def _initialize_augmented_state_space(
        self, rume, unknown_params, num_realizations, initial_values, rng
    ):
        R = num_realizations
        N = rume.scope.nodes
        C = rume.ipm.num_compartments
        current_param_values = {
            k: np.zeros(shape=(R, N), dtype=np.float64) for k in unknown_params.keys()
        }
        for i_realization in range(num_realizations):
            for k in current_param_values.keys():
                if isinstance(unknown_params[k].prior, Prior):
                    current_param_values[k][i_realization, ...] = unknown_params[
                        k
                    ].prior.sample(size=(N,), rng=rng)
                else:
                    current_param_values[k][i_realization, ...] = unknown_params[
                        k
                    ].prior[i_realization, ...]

        current_compartment_values = np.zeros(shape=(R, N, C), dtype=np.int64)
        if initial_values is not None:
            current_compartment_values = initial_values
        else:
            params_temp = {**rume.params}
            for i_realization in range(num_realizations):
                for k in unknown_params.keys():
                    params_temp[k] = current_param_values[k][i_realization, ...]
                rume_temp = dataclasses.replace(rume, params=params_temp)
                data = rume_temp.evaluate_params(rng=rng)
                current_compartment_values[i_realization, ...] = rume.initialize(
                    data=data, rng=rng
                )
        return current_compartment_values, current_param_values

    def _create_sim_rume(
        self,
        rume,
        realization_index,
        unknown_params,
        current_param_values,
        current_compartment_values,
        override_time_frame,
        rng,
    ):
        # Add unknown params to the RUME
        params_temp = {**rume.params}
        for k in unknown_params.keys():
            params_temp[k] = unknown_params[k].dynamics.with_initial(
                current_param_values[k][realization_index, ...]
            )
        rume_temp = dataclasses.replace(
            rume, params=params_temp, time_frame=override_time_frame
        )

        # Evaluate the params.
        data = rume_temp.evaluate_params(rng=rng)
        rume_temp = dataclasses.replace(
            rume_temp,
            params={k.to_pattern(): v for k, v in data.to_dict().items()},
        )

        # Add the compartment values.
        strata_temp = [
            dataclasses.replace(
                stratum,
                init=Explicit(
                    initials=current_compartment_values[realization_index, ...][
                        ..., rume.compartment_mask[stratum.name]
                    ]
                ),
            )
            for stratum in rume.strata
        ]
        rume_temp = dataclasses.replace(rume_temp, strata=strata_temp)
        return rume_temp, data

    def _normalize_log_weights(self, log_weights):
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
        # Calculating the normalized weights from the unnormalized log weights is
        # essentiallty calculating the soft max of the of the log weights, so we
        # borrow a number of techniques here.
        shifted_log_weights = log_weights - np.max(log_weights, keepdims=True)
        underflow_lower_bound = -(10**2)
        clipped_log_weights = np.clip(
            shifted_log_weights, a_min=underflow_lower_bound, a_max=None
        )
        weights = np.exp(clipped_log_weights)
        weights = weights / np.sum(weights, keepdims=True)
        return weights

    def _observation_time_frames(self, rume, time_agg):
        """
        Find the sub time frames which correspond to each observation.

        Parameters
        ----------
        rume :
            The rume of the simulation.
        time_agg :
            The time aggregation.


        Returns
        -------
        :
            The list of time frames and the corresponding labels.
        """
        dim = Dim(nodes=1, days=rume.time_frame.days, tau_steps=rume.num_tau_steps)
        dates = np.repeat(rume.time_frame.to_numpy(), rume.num_tau_steps)
        ticks = np.arange(rume.num_ticks)
        labels = time_agg.grouping.map(dim, ticks, dates)

        # It is important that order is preserved.
        _, label_idx = np.unique(labels, return_index=True)
        label_idx = np.sort(label_idx)
        values = labels[label_idx]

        time_frames = []
        for i in range(len(label_idx) - 1):
            start_date = datetime.date.fromisoformat(str(dates[label_idx[i]]))
            end_date = datetime.date.fromisoformat(str(dates[label_idx[i + 1]]))
            time_frame = TimeFrame.rangex(
                start_date=start_date, end_date_exclusive=end_date
            )
            time_frames.append(time_frame)

        start_date = datetime.date.fromisoformat(str(dates[label_idx[-1]]))
        end_date = datetime.date.fromisoformat(str(dates[-1]))
        time_frames.append(TimeFrame.range(start_date=start_date, end_date=end_date))

        return time_frames, values

    def _multinomial_resampling(self, weights, rng):
        num_realizations = len(weights)
        resampled_idx = rng.choice(
            np.arange(num_realizations),
            size=num_realizations,
            replace=True,
            p=weights,
        )

        resampled_idx = np.sort(resampled_idx)
        return resampled_idx

    def _systematic_resampling(self, weights, rng):
        num_realizations = len(weights)
        c = np.cumsum(weights)
        u = rng.uniform(0, 1)
        resampled_idx = []
        for i_realization in range(num_realizations):
            u_i = (u + i_realization) / num_realizations
            j = np.where(c >= u_i)[0][0]
            resampled_idx.append(j)
        return np.array(resampled_idx)

    def _calculate_prediction():
        """
        Computes the predicted observation.
        """

    def _calculate_block_log_likelihoods():
        """
        Computes the likelihood of the observation given the predicted values for each
        localization block.
        """


class ForecastSimulator(ParticleFilterSimulator):
    def __init__(self, config, duration):
        self.config = config
        self.duration = duration

    def run(self, rng):
        return self._run_forecast(
            **self.config.configure(),
            duration=self.duration,
            rng=rng,
        )

    def _run_forecast(
        self,
        rume,
        num_realizations,
        initial_values,
        param_values,
        unknown_params,
        duration,
        rng,
    ):
        """

        Parameters
        ----------
        rume :
            The RUME.
        num_realizations :
            The number of realizations.
        initial_values :
            The initial compartment values across each realization. The first dimension
            is the realization dimension.
        param_values :
            The parameter values. Overrides values which appear in unknown_params.
        unknown_params :
            The estimated parameter values.
        duration :
            The duration of the forecast.
        observations:
            The observations.
        rng :
            The random number generator.
        Returns
        -------
        :
            The forecast output.
        """
        assert param_values is None
        unknown_params = {NamePattern.of(k): v for k, v in unknown_params.items()}
        original_rume = rume
        sim_time_frame = TimeFrame.of(
            original_rume.time_frame.start_date, duration_days=duration
        )
        rume = dataclasses.replace(
            original_rume,
            time_frame=sim_time_frame,
        )

        context = Context.of(scope=rume.scope, time_frame=rume.time_frame, rng=rng)
        new_params = {}
        for name, param in rume.params.items():
            if isinstance(param, ADRIO):
                new_params[name] = param.with_context_internal(context).evaluate()
            else:
                new_params[name] = param

        rume = dataclasses.replace(
            rume,
            params=new_params,
        )

        # Dimension of the system.
        days = rume.time_frame.days
        taus = rume.num_tau_steps
        R = num_realizations
        T = days
        S = days * taus
        N = rume.scope.nodes
        C = rume.ipm.num_compartments
        E = rume.ipm.num_events

        # Allocate return values.
        initial = np.zeros(shape=(R, N, C), dtype=np.int64)
        compartments = np.zeros(shape=(R, S, N, C), dtype=np.int64)
        events = np.zeros(shape=(R, S, N, E), dtype=np.int64)
        estimated_params = {
            k: np.zeros(shape=(R, T, N), dtype=np.float64)
            for k in unknown_params.keys()
        }

        # Initialize the initial compartment and parameter values.
        current_compartment_values, current_param_values = (
            self._initialize_augmented_state_space(
                rume, unknown_params, num_realizations, initial_values, rng
            )
        )
        initial = current_compartment_values

        for j_realization in range(num_realizations):
            rume_temp, data = self._create_sim_rume(
                rume,
                j_realization,
                unknown_params,
                current_param_values,
                current_compartment_values,
                sim_time_frame,
                rng,
            )

            sim = BasicSimulator(rume_temp)
            out_temp = sim.run(rng_factory=lambda: rng)

            compartments[j_realization, ...] = out_temp.compartments
            events[j_realization, ...] = out_temp.events
            current_compartment_values[j_realization, ...] = out_temp.compartments[
                -1, ...
            ]

            for k in unknown_params.keys():
                estimated_params[k][j_realization, ...] = data.get_raw(k)
                current_param_values[k][j_realization, ...] = data.get_raw(k)[-1, ...]

        return SimpleNamespace(
            rume=original_rume,
            sim_rume=rume,
            duration=duration,
            initial=initial,
            compartments=compartments,
            events=events,
            num_realizations=num_realizations,
            estimated_params=estimated_params,
            unknown_params=unknown_params,
            final_compartment_values=current_compartment_values,
            final_param_values=current_param_values,
        )


class MultiRealizationOutput:
    simulator: MultiRealizationSimulator
    final_compartment_values: NDArray
    final_parameter_values: Sequence[NDArray]
    compartments: NDArray | None
    events: NDArray | None
    initial: NDArray | None


class ParticleFilterOutput(MultiRealizationOutput): ...


class ModelLink(SimpleNamespace): ...
