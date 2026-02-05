import dataclasses
import datetime
from abc import abstractmethod
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import List, Mapping, Self

import numpy as np
from numpy.typing import NDArray

from epymorph.adrio.adrio import ADRIO
from epymorph.attribute import NamePattern
from epymorph.compartment_model import QuantityStrategy
from epymorph.forecasting.dynamic_params import ParamFunctionDynamics, Prior
from epymorph.forecasting.likelihood import Likelihood
from epymorph.geography.scope import GeoStrategy
from epymorph.initializer import Explicit
from epymorph.rume import RUME
from epymorph.simulation import Context
from epymorph.simulator.basic.basic_simulator import BasicSimulator
from epymorph.time import Dim, TimeFrame, TimeStrategy
from epymorph.tools.data import munge


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

    compartments: NDArray
    """
    An array of shape (R, S, N, C) where R is the number of realizations, S is
    the number of tau steps, N is the number of nodes, and C is the number of
    compartments. The interpretation of this output depends on the simulator
    which produced it. For example, each realization could represent a full trajectory
    of the model. Alternatively it could represent a sampled estimate of the compartment
    values with no guarantees of the temporal structure.
    """

    events: NDArray
    """
    An array of shape (R, S, N, E) where R is the number of realizations, S is
    the number of tau steps, N is the number of nodes, and E is the number of
    events. The interpretation of this output depends on the simulator
    which produced it. For example, each realization could represent a full trajectory
    of the model. Alternatively it could represent a sampled estimate of the compartment
    values with no guarantees of the temporal structure.
    """

    initial: NDArray
    """
    An array of shape (R, N, C) where R is the number of realizations, N is the
    number of nodes, and C is the number of compartments. The interpretation of this
    output depends on the simulator which produced it.
    """

    estimated_params: Mapping[NamePattern, NDArray]
    """
    A dictionary where the keys are unknown parameters and the values are
    arrays of shape (R, T, N) where R is the number of realizations, T is the number of
    days and N is the number of nodes. Each realization of a parameter is an array of
    parameter values suitable for intializing another simulation. The precise
    interpretation of the array depends on the simulator used to produce the output.
    """

    @property
    def rume(self) -> RUME:
        return self.simulator.rume

    @property
    def unknown_params(self) -> Mapping[NamePattern, UnknownParam]:
        return self.simulator.unknown_params

    @property
    def num_realizations(self) -> int:
        return self.simulator.num_realizations

    @property
    def initial_values(self) -> NDArray:
        return self.simulator.initial_values


@dataclass(frozen=True)
class PipelineConfig:
    """
    Contains the basic information needed to initialize a PipelineSimulator in order to
    facilitate running a pipeline. Some simulators may require additional information.
    """

    rume: RUME
    """
    The RUME used by the simulator.
    """

    num_realizations: int
    """
    The number of realizations of the simulator.
    """

    initial_values: NDArray | None
    """
    The optional array of initial compartment values of the simulator. It has shape
    (R, N, C) where R is the number of realizations, N is the number of nodes,
    and C is the number of compartments.
    """

    unknown_params: Mapping[NamePattern, UnknownParam]
    """
    The dictionary of unknown paramters of the simulator.
    """

    @classmethod
    def from_rume(
        cls,
        rume,
        num_realizations,
        initial_values=None,
        unknown_params={},
    ):
        return cls(
            rume=rume,
            num_realizations=num_realizations,
            initial_values=initial_values,
            unknown_params={NamePattern.of(k): v for k, v in unknown_params.items()},
        )

    @classmethod
    def from_output(
        cls, output: PipelineOutput, extend_duration: int, override_dynamics={}
    ) -> Self:
        new_time_frame = TimeFrame.of(
            start_date=output.rume.time_frame.end_date + datetime.timedelta(1),
            duration_days=extend_duration,
        )
        rume = replace(output.rume, time_frame=new_time_frame)
        num_realizations = output.num_realizations
        initial_values = output.final_compartment_values
        unknown_params = {
            k: UnknownParam(prior=output.final_param_values[k], dynamics=v.dynamics)
            for k, v in output.unknown_params.items()
        }
        override_dynamics = {NamePattern.of(k): v for k, v in override_dynamics.items()}
        for k, v in override_dynamics.items():
            name = NamePattern.of(k)
            unknown_params[name] = UnknownParam(
                prior=unknown_params[name].prior, dynamics=v
            )
        return cls(
            rume=rume,
            num_realizations=num_realizations,
            initial_values=initial_values,
            unknown_params=unknown_params,
        )


class PipelineSimulator:
    """
    A base class for multi-realization simulations.
    """

    rume: RUME
    """
    The RUME containing the IPM, MM, scope, time_frame, and the fixed params for each
    realization.
    """

    num_realizations: int
    """
    The number of realzations.
    """

    initial_values: NDArray
    """
    An optional array of initial compartment values of shape (R, N, C) where R is the
    number of realizations, N is the number of nodes, and C is the number of
    compartments. If not specified, the initializer contained in the RUME will be used
    to generate a single initial condition which will be used for all realizations.
    """

    unknown_params: Mapping[NamePattern, UnknownParam]
    """
    An array of unknown parameters which will vary across realizations.
    """

    @abstractmethod
    def run(self, rng: np.random.Generator) -> PipelineOutput:
        """
        Run the simulation.
        """


def _initialize_augmented_state_space(
    rume, unknown_params, num_realizations, initial_values, rng
):
    R = num_realizations  # noqa: N806
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
                current_param_values[k][i_realization, ...] = unknown_params[k].prior[
                    i_realization, ...
                ]

    current_compartment_values = np.zeros(shape=(R, N, C), dtype=np.int64)
    if initial_values is not None:
        current_compartment_values = initial_values.copy()
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


@dataclass(frozen=True)
class _SimulateRealizationsResult:
    compartments: NDArray
    events: NDArray
    estimated_params: Mapping[NamePattern, NDArray]
    predictions: List[NDArray]


def _simulate_realizations(
    rume_template: RUME,
    override_time_frame: TimeFrame,
    num_realizations: int,
    initial_values: NDArray,
    unknown_params: Mapping[NamePattern, UnknownParam],
    param_values: Mapping[NamePattern, NDArray],
    geo: GeoStrategy,
    time: TimeStrategy,
    quantity: QuantityStrategy,
    rng: np.random.Generator,
) -> _SimulateRealizationsResult:
    days = override_time_frame.days
    taus = rume_template.num_tau_steps
    R = num_realizations  # noqa: N806
    T = days
    S = days * taus
    N = rume_template.scope.nodes
    C = rume_template.ipm.num_compartments
    E = rume_template.ipm.num_events

    compartments = np.zeros(shape=(R, S, N, C), dtype=np.int64)
    events = np.zeros(shape=(R, S, N, E), dtype=np.int64)
    estimated_params = {
        k: np.zeros(shape=(R, T, N), dtype=np.float64) for k in unknown_params.keys()
    }

    current_predictions = []
    for i_realization in range(num_realizations):
        params_temp = {**rume_template.params}
        for name in unknown_params.keys():
            params_temp[name] = unknown_params[name].dynamics.with_initial(
                param_values[name][i_realization, ...]
            )
        rume_temp = dataclasses.replace(
            rume_template, params=params_temp, time_frame=override_time_frame
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
                    initials=initial_values[i_realization, ...][
                        ..., rume_template.compartment_mask[stratum.name]
                    ]
                ),
            )
            for stratum in rume_template.strata
        ]
        rume_temp = dataclasses.replace(rume_temp, strata=strata_temp)

        sim = BasicSimulator(rume_temp)
        out_temp = sim.run(rng_factory=lambda: rng)

        temp_time_agg = out_temp.rume.time_frame.select.all().agg()
        new_time_agg = dataclasses.replace(
            temp_time_agg,
            aggregation=time.aggregation,
        )

        predicted_df = munge(
            out_temp,
            geo=geo,
            time=new_time_agg,
            quantity=quantity,
        )

        prediction = predicted_df.drop(["geo", "time"], axis=1)

        current_predictions.append(prediction.to_numpy())

        compartments[i_realization, ...] = out_temp.compartments
        events[i_realization, ...] = out_temp.events

        for k in unknown_params.keys():
            estimated_params[k][i_realization, ...] = data.get_raw(k)

    return _SimulateRealizationsResult(
        compartments=compartments,
        events=events,
        estimated_params=estimated_params,
        predictions=current_predictions,
    )


@dataclass(frozen=True)
class ParticleFilterOutput(PipelineOutput):
    """
    Output object for the particle filter which contains additional output and
    diagnostic information.
    """

    posterior_values: NDArray | None
    """
    The posterior estimate of the observed data. The first dimension of the array is the
    number of realizations, the second dimension is the number of observations. The
    remaining dimensions depend on the shape of the observed data. These values are
    constructed from resampling the observed data in the same way the compartment and
    parameter values are resampled.
    """

    effective_sample_size: NDArray | None
    """
    The effective sample size for each observation and each node.
    """


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
    ):
        self.rume = config.rume
        self.num_realizations = config.num_realizations
        self.unknown_params = config.unknown_params
        self.initial_values = config.initial_values
        self.observations = observations

    def run(self, rng):
        output = self._run_particle_filter(
            rume=self.rume,
            num_realizations=self.num_realizations,
            initial_values=self.initial_values,
            unknown_params=self.unknown_params,
            observations=self.observations,
            rng=rng,
        )
        return ParticleFilterOutput(
            simulator=self,
            final_compartment_values=output.final_compartment_values,
            final_param_values=output.final_param_values,
            compartments=output.compartments,
            events=output.events,
            initial=output.initial,
            posterior_values=output.posterior_values,
            effective_sample_size=output.effective_sample_size,
            estimated_params=output.estimated_params,
        )

    def _run_particle_filter(
        self,
        rume,
        num_realizations,
        initial_values,
        unknown_params,
        observations,
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
        initial = np.zeros(shape=(R, N, C), dtype=np.int64)
        compartments = np.zeros(shape=(R, S, N, C), dtype=np.int64)
        events = np.zeros(shape=(R, S, N, E), dtype=np.int64)
        estimated_params = {
            k: np.zeros(shape=(R, T, N), dtype=np.float64)
            for k in unknown_params.keys()
        }

        # Initialize the initial compartment and parameter values. Note, the values
        # within current_compartment_values will be modified in place!
        current_compartment_values, current_param_values = (
            _initialize_augmented_state_space(
                rume, unknown_params, num_realizations, initial_values, rng
            )
        )

        initial = current_compartment_values.copy()

        # Determine the number of observations and their associated time frames.
        time_frames, labels = self._observation_time_frames(
            rume, observations.model_link.time
        )
        resampling_indices = np.zeros(shape=(R, len(time_frames), N), dtype=np.int32)
        resampling_days = np.zeros(shape=(len(time_frames),), dtype=np.int32)
        resampling_steps = np.zeros(shape=(len(time_frames),), dtype=np.int32)
        effective_sample_size = np.zeros(shape=(len(time_frames), N), dtype=np.float64)

        context = Context.of(
            scope=rume.scope,
            time_frame=rume.time_frame,
            ipm=rume.ipm,
            rng=np.random.default_rng(0),
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
                )
            )

            # current_predictions = []

            result = _simulate_realizations(
                rume_template=rume,
                override_time_frame=time_frame,
                num_realizations=num_realizations,
                initial_values=current_compartment_values,
                unknown_params=unknown_params,
                param_values=current_param_values,
                geo=observations.model_link.geo,
                time=observations.model_link.time,
                quantity=observations.model_link.quantity,
                rng=rng,
            )

            current_compartment_values = result.compartments[:, -1, ...]
            current_predictions = result.predictions

            compartments[:, step_idx:step_right, ...] = result.compartments
            events[:, step_idx:step_right, ...] = result.events

            for k in unknown_params.keys():
                estimated_params[k][:, day_idx:day_right, ...] = (
                    result.estimated_params[k]
                )
                current_param_values[k] = result.estimated_params[k][:, -1, ...].copy()

            weights = np.zeros(num_realizations, dtype=np.float64)
            log_weights_by_node = np.zeros((num_realizations, N), dtype=np.float64)
            for j_realization in range(num_realizations):
                if observation_array.size > 0:
                    node_log_likelihoods = observations.likelihood.compute_log(
                        observation_array["value"][0, :],
                        current_predictions[j_realization][:, 0],
                    )
                else:
                    node_log_likelihoods = np.zeros(N)

                log_weights_by_node[j_realization, ...] = node_log_likelihoods

            predicted_values.append(np.array(current_predictions))

            posterior_value = predicted_values[-1].copy()

            prior_compartment_values = current_compartment_values.copy()

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
                    prior_compartment_values[resampled_idx, i_node, ...]
                )
                for k in current_param_values.keys():
                    current_param_values[k][orig_idx, i_node] = current_param_values[k][
                        resampled_idx, i_node
                    ]

                posterior_value[orig_idx, i_node, ...] = posterior_value[
                    resampled_idx, i_node, ...
                ]

                resampling_indices[:, i_observation, i_node] = resampled_idx[:, 0]
                effective_sample_size[i_observation, i_node] = 1 / np.sum(weights**2)

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
            effective_sample_size=effective_sample_size,
        )

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

        # Shifting the log weights is equivalent to multiplying the non-log weights by a
        # constant. This helps avoid underflow issues in the non-log weights without
        # affecting the final normalized output.
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


class ForecastSimulator(PipelineSimulator):
    def __init__(self, config):
        self.rume = config.rume
        self.num_realizations = config.num_realizations
        self.unknown_params = config.unknown_params
        self.initial_values = config.initial_values

    def run(self, rng) -> PipelineOutput:
        rume = self.rume
        num_realizations = self.num_realizations
        initial_values = self.initial_values
        unknown_params = self.unknown_params
        rng = rng
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

        # Initialize the initial compartment and parameter values.
        current_compartment_values, current_param_values = (
            _initialize_augmented_state_space(
                rume, unknown_params, num_realizations, initial_values, rng
            )
        )
        initial = current_compartment_values

        result = _simulate_realizations(
            rume_template=rume,
            override_time_frame=rume.time_frame,
            num_realizations=num_realizations,
            initial_values=current_compartment_values,
            unknown_params=unknown_params,
            param_values=current_param_values,
            geo=rume.scope.select.all(),
            time=rume.time_frame.select.all(),
            quantity=rume.ipm.select.all(),
            rng=rng,
        )

        return PipelineOutput(
            simulator=self,
            final_compartment_values=result.compartments[:, -1, ...],
            final_param_values={
                k: result.estimated_params[k][:, -1, ...] for k in unknown_params.keys()
            },
            compartments=result.compartments,
            events=result.events,
            initial=initial,
            estimated_params=result.estimated_params,
        )
