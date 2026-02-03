import dataclasses
import datetime
from abc import abstractmethod
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Mapping

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from epymorph.adrio.adrio import ADRIO
from epymorph.attribute import NamePattern
from epymorph.compartment_model import QuantityStrategy, compartment
from epymorph.forecasting.dynamic_params import ParamFunctionDynamics, Prior
from epymorph.forecasting.likelihood import Likelihood,Gaussian
from epymorph.geography.scope import GeoStrategy
from epymorph.initializer import Explicit
from epymorph.parameter_fitting.utils import observations
from epymorph.rume import RUME
from epymorph.simulation import Context
from epymorph.simulator.basic.basic_simulator import BasicSimulator
from epymorph.time import Dim, TimeFrame, TimeStrategy
from epymorph.tools.data import munge

from time import perf_counter

@dataclass(frozen=True)
class UnknownParam:
    prior: NDArray | Prior
    dynamics: ParamFunctionDynamics


@dataclass(frozen=True)
class PipelineOutput:
    """
    The output of a `PipelineSimulator` run. This includes the minimum information
    needed to initialize another `PipelineSimulator`, most frequently using the
    `FromOutput` config. Due to the inherently high memory usage of multi-dimensional
    arrays, many attributes are optional.
    """

    simulator: "PipelineSimulator"
    """
    The simulator used to generate the output. It contains information like the RUME,
    the (prior) unknown parameters,  the number of realizations, and how the
    compartments were initialized.
    """

    final_compartment_values: NDArray
    """
    An array of shape (R, N, C) where R is the number of realizations, N is the number
    of nodes, and C is the number of compartments. Each realization is an array of
    compartment values suitable for intializing another simulation. The precise
    interpretation of the array depends on the simulator used to produce the output.
    """

    final_param_values: Mapping[NamePattern, NDArray]
    """
    A dictionary where the keys are unknown parameters and the values are of arrays of
    shape (R, N) where R is the number of realizations and N is the number of nodes.
    Each realization of a parameter is an array of parameter values suitable for
    intializing another simulation. The precise interpretation of the array depends on
    the simulator used to produce the output.
    """

    compartments: NDArray | None
    events: NDArray | None
    initial: NDArray | None

    @property
    def rume(self):
        return self.simulator.rume

    @property
    def unknown_params(self):
        return self.simulator.unknown_params

    @property
    def num_realizations(self):
        return self.simulator.num_realizations

    @property
    def initial_values(self):
        return self.simulator.initial_values

    def to_npz(self, file):
        """
        Saves the compartment and parameter values as numpy arrays in a .npz file. Note
        that other information, such as the RUME and the parameter dynamics, are not
        stored and will need to be provided separately upon loading.
        """
        param_names = np.array([str(k) for k in self.final_param_values.keys()])
        np.savez(
            file,
            final_compartment_values=self.final_compartment_values,
            param_names=param_names,
            **{str(k): v for k, v in self.final_param_values.items()},
        )


class PipelineConfig:
    rume: RUME
    num_realizations: int
    initial_values: NDArray | None
    unknown_params: Mapping[NamePattern, UnknownParam]


class PipelineSimulator:
    rume: RUME
    num_realizations: int
    initial_values: NDArray | None
    unknown_params: Mapping[NamePattern, UnknownParam]

    @abstractmethod
    def run(self, rng: np.random.Generator) -> PipelineOutput:
        """
        Run the simulation.
        """


@dataclass(frozen=True)
class ParticleFilterOutput(PipelineOutput):
    posterior_values: NDArray | None
    estimated_params: NDArray | None


class FromRUME(PipelineConfig):
    def __init__(
        self,
        rume,
        num_realizations,
        initial_values=None,
        unknown_params={},
    ):
        self.rume = rume
        self.num_realizations = num_realizations
        self.initial_values = initial_values
        self.unknown_params = {NamePattern.of(k): v for k, v in unknown_params.items()}


class FromOutput(PipelineConfig):
    def __init__(self, output: PipelineOutput, extend_duration, override_dynamics={}):
        override_dynamics = {NamePattern.of(k): v for k, v in override_dynamics.items()}
        new_time_frame = TimeFrame.of(
            start_date=output.rume.time_frame.end_date + datetime.timedelta(1),
            duration_days=extend_duration,
        )
        self.rume = replace(output.rume, time_frame=new_time_frame)
        self.num_realizations = output.num_realizations
        self.initial_values = output.final_compartment_values
        self.unknown_params = {
            k: UnknownParam(prior=output.final_param_values[k], dynamics=v.dynamics)
            for k, v in output.unknown_params.items()
        }
        for k, v in override_dynamics.items():
            name = NamePattern.of(k)
            self.unknown_params[name] = UnknownParam(
                prior=self.unknown_params[name].prior, dynamics=v
            )


class FromNPZ(PipelineConfig):
    def __init__(
        self, file, rume, unknown_params, extend_duration, override_dynamics={}
    ):
        data = np.load(file)
        final_compartment_values = data["final_compartment_values"]
        final_param_values = {
            NamePattern.of(name): data[name] for name in data["param_names"]
        }
        data.close()

        num_realizations = final_compartment_values.shape[0]

        override_dynamics = {NamePattern.of(k): v for k, v in override_dynamics.items()}
        new_time_frame = TimeFrame.of(
            start_date=rume.time_frame.end_date + datetime.timedelta(1),
            duration_days=extend_duration,
        )
        self.rume = replace(rume, time_frame=new_time_frame)
        self.num_realizations = num_realizations
        self.initial_values = final_compartment_values
        self.unknown_params = {
            k: UnknownParam(prior=final_param_values[k], dynamics=v.dynamics)
            for k, v in unknown_params.items()
        }
        for k, v in override_dynamics.items():
            name = NamePattern.of(k)
            self.unknown_params[name] = UnknownParam(
                prior=self.unknown_params[name].prior, dynamics=v
            )


@dataclass(frozen=True)
class ModelLink:
    """
    Contains the information needed to subselect, group, and aggregate model output in
    order to compare it to observed data.
    """

    geo: GeoStrategy
    time: TimeStrategy
    quantity: QuantityStrategy


@dataclass(frozen=True)
class Observations:
    source: ADRIO
    model_link: ModelLink
    likelihood: Likelihood
    strict_labels: bool = False

    def _get_observations_dataframe(self, context):
        if isinstance(self.source, pd.DataFrame):
            return self.source
        elif isinstance(self.source, ADRIO):
            inspect = self.source.with_context_internal(context).inspect()
            if isinstance(inspect.source, pd.DataFrame):
                return inspect.source

    def _get_observations_numpy_array(self, context):
        if isinstance(self.source, np.ndarray):
            return self.source
        elif isinstance(self.source, ADRIO):
            inspect = self.source.with_context_internal(context).inspect()
            return inspect.values


class ParticleFilterSimulator(PipelineSimulator):
    def __init__(
        self,
        config: PipelineConfig,
        observations: Observations,
        save_trajectories=True,
    ):
        self.rume = config.rume
        self.num_realizations = config.num_realizations
        self.unknown_params = config.unknown_params
        self.initial_values = config.initial_values
        self.observations = observations
        self.save_trajectories = save_trajectories

    def run(self, rng):
        t0 = perf_counter()
        output = self._run_particle_filter(
            self.rume,
            self.num_realizations,
            self.initial_values,
            self.unknown_params,
            observations=self.observations,
            rng=rng,
            save_trajectories=self.save_trajectories,
        )
        t1 = perf_counter()

        print(f"\nRuntime with {self.num_realizations} particles and {self.rume.scope.nodes} Nodes is {t1-t0} seconds.")

        return ParticleFilterOutput(
            simulator=self,
            final_compartment_values=output.final_compartment_values,
            final_param_values=output.final_param_values,
            compartments=output.compartments,
            events=output.events,
            initial=output.initial,
            posterior_values=output.posterior_values,
            estimated_params=output.estimated_params
        )

    def _run_particle_filter(
        self,
        rume,
        num_realizations,
        initial_values,
        unknown_params,
        observations,
        rng,
        save_trajectories=True,
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
        unknown_params = {NamePattern.of(k): v for k, v in unknown_params.items()}

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

        initial = None
        compartments = None
        events = None
        estimated_params = None

        if save_trajectories:
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
        if save_trajectories:
            initial = current_compartment_values

        # Determine the number of observations and their associated time frames.
        time_frames, labels = self._observation_time_frames(
            rume, observations.model_link.time
        )
        resampling_indices = np.zeros(shape=(R, len(time_frames), N), dtype=np.int32)
        resampling_days = np.zeros(shape=(len(time_frames),), dtype=np.int32)
        resampling_steps = np.zeros(shape=(len(time_frames),), dtype=np.int32)

        context = Context.of(
            scope=rume.scope,
            time_frame=rume.time_frame,
            ipm=rume.ipm,
            rng=np.random.default_rng(0),
        )
        # observations_df = observations._get_observations_dataframe(context)
        observations_array = observations._get_observations_numpy_array(context)

        observed_values = []
        predicted_values = []
        posterior_values = []

        day_idx = 0
        step_idx = 0
        for i_observation in range(len(time_frames)):
            time_frame = time_frames[i_observation]

            day_right = day_idx + time_frame.days
            step_right = step_idx + time_frame.days * rume.num_tau_steps

            # TODO # self._get_current_observation(...)
            # time_key = observations_df.keys()[0]
            # geo_key = observations_df.keys()[1]

            start = np.datetime64(time_frame.start_date.isoformat())
            end = np.datetime64(time_frame.end_date.isoformat())
            observation_array = observations_array[
                (observations_array["date"][:, 0] >= start)
                & (observations_array["date"][:, 0] <= end),
                :,
            ]

            print(  # noqa: T201
                (
                    f"Observation: {i_observation}, "
                    f"Label: {labels[i_observation]}, "
                    f"Time Frame: {time_frame}"
                ),
                end = '\r'
            )

            current_predictions = []
            weights = np.zeros(num_realizations, dtype=np.float64)
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

                if observation_array.size > 0:
                    node_log_likelihoods = observations.likelihood.compute_log(
                        observation_array["value"][0, :],
                        prediction.to_numpy()[:, 0],
                    )
                else:
                    node_log_likelihoods = np.zeros(N)

                log_weights_by_node[j_realization, ...] = node_log_likelihoods

                if save_trajectories:
                    compartments[j_realization, step_idx:step_right, ...] = (
                        out_temp.compartments
                    )
                    events[j_realization, step_idx:step_right, ...] = out_temp.events
                    current_compartment_values[j_realization, ...] = (
                        out_temp.compartments[-1, ...]
                    )

                for k in unknown_params.keys():
                    if save_trajectories:
                        estimated_params[k][j_realization, day_idx:day_right, ...] = (
                            data.get_raw(k)
                        )
                    current_param_values[k][j_realization, ...] = data.get_raw(k)[
                        -1, ...
                    ]

            predicted_values.append(np.array(current_predictions))

            posterior_value = predicted_values[-1].copy()

            # Hard code localization
            for i_node in range(N):
                if (
                    observation_array.size > 0
                    and np.ma.is_masked(observation_array["value"])
                    and observation_array.mask["value"][0, i_node]
                ):
                    weights = 1 / num_realizations * np.ones((num_realizations,))
                else:
                    weights = self._normalize_log_weights(
                        log_weights_by_node[..., i_node]
                    )
                orig_idx = np.arange(0, num_realizations)[:, np.newaxis]
                resampled_idx = self._systematic_resampling(weights, rng)[:, np.newaxis]
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

            resampling_steps[i_observation] = step_right
            resampling_days[i_observation] = day_right

            day_idx = day_right
            step_idx = step_right

        return SimpleNamespace(
            rume=rume,
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

class ForecastSimulator(ParticleFilterSimulator):
    def __init__(self, config, save_trajectories=True):
        self.rume = config.rume
        self.num_realizations = config.num_realizations
        self.unknown_params = config.unknown_params
        self.initial_values = config.initial_values
        self.save_trajectories = save_trajectories

    def run(self, rng) -> PipelineOutput:
        output = self._run_forecast(
            rume=self.rume,
            num_realizations=self.num_realizations,
            initial_values=self.initial_values,
            unknown_params=self.unknown_params,
            rng=rng,
            save_trajectories=self.save_trajectories,
        )
        return PipelineOutput(
            simulator=self,
            final_compartment_values=output.final_compartment_values,
            final_param_values=output.final_param_values,
            compartments=output.compartments,
            events=output.events,
            initial=output.initial,
        )

    def _run_forecast(
        self,
        rume,
        num_realizations,
        initial_values,
        unknown_params,
        rng,
        save_trajectories=True,
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
        unknown_params = {NamePattern.of(k): v for k, v in unknown_params.items()}
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
        initial = None
        compartments = None
        events = None
        estimated_params = None

        if save_trajectories:
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
                rume.time_frame,
                rng,
            )

            sim = BasicSimulator(rume_temp)
            out_temp = sim.run(rng_factory=lambda: rng)

            current_compartment_values[j_realization, ...] = out_temp.compartments[
                -1, ...
            ]

            for k in unknown_params.keys():
                current_param_values[k][j_realization, ...] = data.get_raw(k)[-1, ...]

            if save_trajectories:
                compartments[j_realization, ...] = out_temp.compartments
                events[j_realization, ...] = out_temp.events

                for k in unknown_params.keys():
                    estimated_params[k][j_realization, ...] = data.get_raw(k)

        return SimpleNamespace(
            rume=rume,
            sim_rume=rume,
            initial=initial,
            compartments=compartments,
            events=events,
            num_realizations=num_realizations,
            estimated_params=estimated_params,
            unknown_params=unknown_params,
            final_compartment_values=current_compartment_values,
            final_param_values=current_param_values,
        )

class EnsembleKalmanFilterSimulator(PipelineSimulator):
    def __init__(
        self,
        config: PipelineConfig,
        observations: Observations,
        save_trajectories=True,
    ):
        self.rume = config.rume
        self.num_realizations = config.num_realizations
        self.unknown_params = config.unknown_params
        self.initial_values = config.initial_values
        self.observations = observations
        self.save_trajectories = save_trajectories

    def ensure_col(self,arr): 
        if arr.ndim == 1: 
            return arr[...,np.newaxis]
        return arr

    def run(self, rng):
        t0 = perf_counter()
        output = self._run_ensemble_kalman_filter(
            self.rume,
            self.num_realizations,
            self.initial_values,
            self.unknown_params,
            observations=self.observations,
            rng=rng,
            save_trajectories=self.save_trajectories,
        )
        t1 = perf_counter()
        print(f"\nRuntime with {self.num_realizations} members and {self.rume.scope.nodes} Nodes is {np.round(t1-t0,2)} seconds.")

        return ParticleFilterOutput(
            simulator=self,
            final_compartment_values=output.final_compartment_values,
            final_param_values=output.final_param_values,
            compartments=output.compartments,
            events=output.events,
            initial=output.initial,
            posterior_values=output.posterior_values,
            estimated_params=output.estimated_params,
        )

    def simulation(self,
                    rume,observations,
                    num_realizations,
                    unknown_params,
                    current_param_values,
                    current_compartment_values,
                    time_frame,
                    rng):

        """

        Parameters
        ----------
        rume :
            The RUME.
        num_realizations :
            The number of realizations.
        unknown_params :
            The estimated parameter names and values.
        observations:
            The observations object, encompassing the model link and aggregator.
        time_frame :
            The TimeFrame corresponding to the current data point
        current_param_values : 
            The current values of the unknown parameters, used for the 
            forward model. 
        current_compartment_values : 
            The initial condition corresponding to each realization. 
        rng :
            The random number generator.
        Returns
        -------
        :
            The simulation output up to the next data point for each realization.
        """

        ###Array initialization
        compartment_update = np.zeros((num_realizations,
                                       rume.num_tau_steps * time_frame.duration_days,
                                       rume.scope.nodes,rume.ipm.num_compartments),
                                       dtype = np.float64)

        events_update = np.zeros((num_realizations,
                                       rume.num_tau_steps * time_frame.duration_days,
                                       rume.scope.nodes,rume.ipm.num_events),
                                       dtype = np.float64)

        params_update = {
                k: np.zeros(shape=(num_realizations, time_frame.duration_days, rume.scope.nodes), 
                            dtype=np.float64)
                for k in unknown_params.keys()
            }

        ### Loop over the realizations, run the forward model for each
        current_predictions = []
        for j_realization in range(num_realizations):
            rume_temp, data = self._create_sim_rume(
                        rume,
                        j_realization,
                        unknown_params,
                        current_param_values,
                        current_compartment_values,
                        time_frame,
                        rng
                    )

            sim = BasicSimulator(rume_temp)
            out_temp = sim.run(rng_factory=lambda: rng)

            temp_time_agg = out_temp.rume.time_frame.select.all().agg()
            new_time_agg = dataclasses.replace(
                temp_time_agg,
                aggregation=observations.model_link.time.aggregation,
            )

            ### Apply our aggregators so we get the predicted observation
            predicted_df = munge(
                out_temp,
                geo=observations.model_link.geo,
                time=new_time_agg,
                quantity=observations.model_link.quantity,
            )

            prediction = predicted_df.drop(["geo", "time"], axis=1).to_numpy()
            current_predictions.append(prediction)

            ###Update the compartments,events and parameters
            compartment_update[j_realization,...] = out_temp.compartments
            events_update[j_realization,...] = out_temp.events

            for k in unknown_params.keys():
                params_update[k][j_realization,...] = data.get_raw(k)


        return current_predictions,compartment_update,events_update,params_update 

    def update(self,
               current_compartment_values,
               current_param_values,
               pred_observations,
               real_observations,
               rng): 

        """

        Parameters
        ----------
        current_compartment_values : 
            The current values of the compartments, the leading index is num_realizations.
        current_param_values : 
            The current values of the parameters, a dictionary where keys are parameter names
            and the values are arrays with leading axis of num_realizations. 
        pred_observations :
            The predicted observations corresponding to each realization, a numpy array 
            with leading axis num_realizations. 
        real_observations: 
            The actual observation data, currently one observation for each node. 
        rng :
            The random number generator.
        Returns
        -------
        :
            The forecast output.
        """

        #Copy current_compartment_values and 
        #current_param_values to ensure they are never edited

        compartment_values = current_compartment_values.copy()
        param_values = {k: v.copy() for k, v in current_param_values.items()}

        N = self.rume.scope.nodes
        C = self.rume.ipm.num_compartments

        r_obs = np.array([[0.]])
        if isinstance(self.observations.likelihood,Gaussian):
            r_obs = np.array([[self.observations.likelihood.variance]])

        for i_node in range(N): 
            node_ensemble = compartment_values[:,i_node].reshape(self.num_realizations,-1)
            for k in param_values.keys(): 
                node_ensemble = np.concatenate((node_ensemble,param_values[k][:,i_node,np.newaxis]),axis = -1)

            node_pred_observations = self.ensure_col(pred_observations)[:,i_node]

            ###Compute useful statistics and matrices
            mean = np.mean(node_ensemble,axis = 0)
            state_perturbation_matrix = 1/np.sqrt(self.num_realizations - 1) * (node_ensemble - mean)

            obs_mean = np.mean(node_pred_observations,axis = 0)
            obs_perturbation_matrix = 1/np.sqrt(self.num_realizations - 1) * (node_pred_observations - obs_mean)

            S_inv = np.linalg.pinv(obs_perturbation_matrix.T @ obs_perturbation_matrix + r_obs)
            cross_cov = state_perturbation_matrix.T @ obs_perturbation_matrix
            cross_cov = cross_cov[...,np.newaxis]

            ###Stochastic noise to match KF equations statistically
            node_perturbed_observations = np.full((self.num_realizations,
                                              *real_observations[i_node].shape),
                                             real_observations[i_node]).astype(np.float64)

            node_perturbed_observations = self.observations.likelihood.sample(rng,node_perturbed_observations)


            ###Update loop
            for i_member in range(self.num_realizations): 

                innovation = node_perturbed_observations[i_member] - node_pred_observations[i_member]
                innovation = innovation[...,np.newaxis]

                alpha = S_inv @ innovation

                member_post = node_ensemble[i_member,:] + cross_cov @ alpha
                compartment_values[i_member,i_node,:] = np.rint(np.clip(member_post[:C],a_min = 0.,a_max = None))
                
                for i_param,k in enumerate(param_values.keys()): 
                    param_values[k][i_member,i_node] = member_post[C+i_param]

        return compartment_values,param_values

    # def update(self,
    #            current_compartment_values,
    #            current_param_values,
    #            pred_observations,
    #            real_observations,
    #            rng): 

    #     """

    #     Parameters
    #     ----------
    #     current_compartment_values : 
    #         The current values of the compartments, the leading index is num_realizations.
    #     current_param_values : 
    #         The current values of the parameters, a dictionary where keys are parameter names
    #         and the values are arrays with leading axis of num_realizations. 
    #     pred_observations :
    #         The predicted observations corresponding to each realization, a numpy array 
    #         with leading axis num_realizations. 
    #     real_observations: 
    #         The actual observation data, currently one observation for each node. 
    #     rng :
    #         The random number generator.
    #     Returns
    #     -------
    #     :
    #         The forecast output.
    #     """

    #     #Copy current_compartment_values and 
    #     #current_param_values to ensure they are never edited

    #     compartment_values = current_compartment_values.copy()
    #     param_values = {k: v.copy() for k, v in current_param_values.items()}

    #     N = self.rume.scope.nodes
    #     C = self.rume.ipm.num_compartments
    #     state_size = N * C

    #     ##State matrix construction
    #     r_obs = self.observations.likelihood.variance * np.eye(N)
    #     ensemble = compartment_values.reshape(self.num_realizations,-1)
    #     for k in param_values.keys(): 
    #         ensemble = np.concatenate((ensemble,param_values[k]),axis = -1)

    #     ###Compute useful statistics and matrices
    #     mean = np.mean(ensemble,axis = 0)
    #     state_perturbation_matrix = 1/np.sqrt(self.num_realizations - 1) * (ensemble - mean)

    #     obs_mean = np.mean(pred_observations,axis = 0)
    #     obs_perturbation_matrix = 1/np.sqrt(self.num_realizations - 1) * (pred_observations - obs_mean)

    #     #cov(obs_pred,obs_pred)
    #     S = obs_perturbation_matrix.T @ obs_perturbation_matrix + r_obs

    #     #Stochastic noise to match KF equations statistically
    #     perturbed_observations = np.full((self.num_realizations,*real_observations.shape),real_observations).astype(np.float64)

    #     noise = rng.normal(0.,scale = np.sqrt(self.observations.likelihood.variance),size = perturbed_observations.shape)
    #     perturbed_observations += noise

    #     #Kalman gain terms
    #     S_inv = np.linalg.pinv(S)
    #     cross_cov = state_perturbation_matrix.T @ obs_perturbation_matrix
    #     cross_cov = cross_cov

    #     #Update Ensemble Members
    #     for i_member in range(self.num_realizations):
    #         innovation = perturbed_observations[i_member,...] - pred_observations[i_member,...]

    #         alpha = S_inv @ innovation

    #         member_post = ensemble[i_member,:] + cross_cov @ alpha

    #         compartment_values[i_member,...] = member_post[:state_size].reshape((N,C))
    #         compartment_values[i_member,...] = np.rint(np.clip(compartment_values[i_member,...],
    #                                                                     a_min = 0., a_max = None))

    #         for i_param,k in enumerate(param_values.keys()): 
    #             i_param_block = int(N + i_param * N)

    #             param_values[k][i_member, ...] = member_post[state_size + i_param * N:state_size+i_param_block]

    #     return compartment_values,param_values

    def _run_ensemble_kalman_filter(
        self,
        rume,
        num_realizations,
        initial_values,
        unknown_params,
        observations,
        rng,
        save_trajectories=True,
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
        unknown_params :
            The estimated parameter dictionary.
        observations:
            The observations.
        rng :
            The random number generator.
        save_trajectories : 
            A boolean indicating whether to save the full trajectories
            of compartments,events, and parameters. Can require large amounts of
            available memory for large models. 
        Returns
        -------
        :
            The ensemble kalman filter output.
        """

        ###Enforcement of Gaussian Likelihood 
        # if not isinstance(self.observations.likelihood,Gaussian): 
        #     raise ValueError("Likelihood must be Gaussian for the EnKF.")

        ###Convert unknown_params to NamePattern type
        unknown_params = {NamePattern.of(k): v for k, v in unknown_params.items()}

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

        initial = None
        compartments = None
        events = None
        estimated_params = None

        if save_trajectories:
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
        if save_trajectories:
            initial = current_compartment_values

        # Determine the number of observations and their associated time frames.
        time_frames, labels = self._observation_time_frames(
            rume, observations.model_link.time
        )

        context = Context.of(
            scope=rume.scope,
            time_frame=rume.time_frame,
            ipm=rume.ipm
        )
        observations_array = observations._get_observations_numpy_array(context)

        observed_values = []
        predicted_values = []
        posterior_values = []

        day_idx = 0
        step_idx = 0

        for i_observation in range(len(time_frames)):
            time_frame = time_frames[i_observation]

            day_right = day_idx + time_frame.days
            step_right = step_idx + time_frame.days * rume.num_tau_steps

            start = np.datetime64(time_frame.start_date.isoformat())
            end = np.datetime64(time_frame.end_date.isoformat())
            observation_array = observations_array[
                (observations_array["date"][:, 0] >= start)
                & (observations_array["date"][:, 0] <= end),
                :,
            ]

            print(  # noqa: T201
                (
                    f"Observation: {i_observation}, "
                    f"Label: {labels[i_observation]}, "
                    f"Time Frame: {time_frame}"
                ),end = "\r"
            )

        ###Simulation Step
            current_predictions,compartment_update,events_update,params_update = self.simulation(rume,
                            observations,
                            num_realizations,
                            unknown_params,
                            current_param_values,
                            current_compartment_values,
                            time_frame,
                            rng)

            #Update if saving trajectories
            if save_trajectories:
                compartments[:, step_idx:step_right, ...] = compartment_update
                events[:, step_idx:step_right, ...] = events_update
                for k in unknown_params.keys():
                    estimated_params[k][:, day_idx:day_right, ...] = params_update[k]

            #Update currents 
            current_compartment_values[:,...] = compartment_update[:,-1,...]
            for k in unknown_params.keys():
                current_param_values[k][:, ...] = params_update[k][:,-1,...]

            pred_observations = np.array(current_predictions).squeeze()
            predicted_values.append(pred_observations)

            posterior_value = predicted_values[-1].copy()

        ###Kalman Update Step###
            real_observation_curr = np.zeros((N,))
            if observation_array.shape[0] != 0:
                real_observation_curr = observation_array["value"][0, :]


            current_compartment_values,current_param_values = self.update(
                        current_compartment_values,
                        current_param_values,
                        pred_observations,
                        real_observation_curr,
                        rng)

            posterior_values.append(posterior_value)

            day_idx = day_right
            step_idx = step_right


        return SimpleNamespace(
            rume=rume,
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
            observed_values=np.array(observed_values)
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
            #Update Ensemble Members
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
