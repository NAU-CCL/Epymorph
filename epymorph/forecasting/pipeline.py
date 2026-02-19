import dataclasses
import datetime
from abc import abstractmethod, ABC
from dataclasses import dataclass, replace
from typing import Generic, List, Mapping, Self, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from epymorph.adrio.adrio import ADRIO, ValueT
from epymorph.attribute import NamePattern
from epymorph.compartment_model import (
    QuantityAggregation,
    QuantitySelection,
)
from epymorph.forecasting.dynamic_params import ParamFunctionDynamics, Prior
from epymorph.forecasting.likelihood import Gaussian, Likelihood
from epymorph.geography.scope import GeoAggregation, GeoSelection
from epymorph.initializer import Explicit
from epymorph.rume import RUME
from epymorph.simulation import Context
from epymorph.simulator.basic.basic_simulator import BasicSimulator
from epymorph.time import (
    Dim,
    GroupKeyType,
    TimeAggregation,
    TimeFrame,
    TimeGrouping,
    TimeSelection,
)
from epymorph.tools.data import munge
from epymorph.util import DateValueType


@dataclass(frozen=True)
class UnknownParam:
    """
    Contains the information for an unknown parameter. An unknown parameter is a
    parameter which can vary across realizations in a multi-realization simulation. Some
    simulators will try to estimate unknown parameters.
    """

    prior: NDArray | Prior
    """
    The prior distribution or initial values of the parameter.
    """

    dynamics: ParamFunctionDynamics
    """
    The dynamics of the parameter dictating how the parameter changes over time.
    """


@dataclass(frozen=True)
class PipelineConfig:
    """
    Contains the basic information needed to initialize a `PipelineSimulator`.
    Some simulators may require additional information provided in their respective
    constructor.
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
        rume: RUME,
        num_realizations: int,
        initial_values: NDArray | None = None,
        unknown_params: Mapping[NamePattern, UnknownParam]
        | Mapping[str, UnknownParam] = {},
    ):
        """
        Creates a PipelineConfig from a RUME. Converts the keys of unknown_params into
        NamePattern's.
        """
        return cls(
            rume=rume,
            num_realizations=num_realizations,
            initial_values=initial_values,
            unknown_params={NamePattern.of(k): v for k, v in unknown_params.items()},
        )

    @classmethod
    def from_output(
        cls,
        output: "PipelineOutput",
        extend_duration: int,
        override_dynamics: Mapping[NamePattern, ParamFunctionDynamics]
        | Mapping[str, ParamFunctionDynamics] = {},
    ) -> Self:
        """
        Creates a PipelineConfig starting from the end of a previous PipelineOutput.
        The RUME, unknown parameters, and initial compartment values are taken from the
        output. The time frame is extended from the end of the previous time frame.

        Parameters
        ----------
        output : PipelineOutput
            The output to extend.
        extend_duration : int
            The number of days to extend the simulation from the end of the previous
            output.
        override_dynamics : Mapping[NamePattern, ParamFunctionDynamics]
            Override the dyanmics of any parameters from the previous output.
        """
        new_time_frame = TimeFrame.of(
            start_date=output.rume.time_frame.end_date + datetime.timedelta(1),
            duration_days=extend_duration,
        )
        rume = replace(output.rume, time_frame=new_time_frame)
        num_realizations = output.num_realizations
        initial_values = output.final_compartments
        unknown_params = {
            k: UnknownParam(prior=output.final_params[k], dynamics=v.dynamics)
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


@dataclass(frozen=True)
class PipelineOutput:
    """
    The output of a `PipelineSimulator` run.
    """

    simulator: "PipelineSimulator"
    """
    The simulator used to generate the output. It contains information like the RUME,
    the (prior) unknown parameters,  the number of realizations, and how the
    compartments were initialized.
    """

    final_compartments: NDArray
    """
    An array of shape (R, N, C) where R is the number of realizations, N is the number
    of nodes, and C is the number of compartments. Each realization is an array of
    compartment values suitable for intializing another simulation. The precise
    interpretation of the array depends on the simulator used to produce the output.
    """

    final_params: Mapping[NamePattern, NDArray]
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
    def initial_values(self) -> NDArray | None:
        return self.simulator.initial_values

    @property
    def dataframe(self) -> pd.DataFrame:
        NP = self.num_realizations
        C = self.rume.ipm.num_compartments
        E = self.rume.ipm.num_events
        N = self.rume.scope.nodes
        S = self.rume.num_ticks
        tau_steps = self.rume.num_tau_steps

        data_np = np.concatenate(
            (self.compartments, self.events),
            axis=3,
        ).reshape((-1, C + E), order="C")

        # Here I'm concatting two DFs sideways so that the index columns come first.
        # Could use insert, but this is nicer.
        return pd.concat(
            (
                # A dataframe for the various indices
                pd.DataFrame(
                    {
                        "realization": np.repeat(np.arange(NP), S * N),
                        "tick": np.tile(np.repeat(np.arange(S), N), NP),
                        "date": np.tile(
                            np.repeat(self.rume.time_frame.to_numpy(), N * tau_steps),
                            NP,
                        ),
                        "node": np.tile(self.rume.scope.node_ids, S * NP),
                    }
                ),
                # A dataframe for the data columns
                pd.DataFrame(
                    data=data_np,
                    columns=[
                        *(c.name.full for c in self.rume.ipm.compartments),
                        *(e.name.full for e in self.rume.ipm.events),
                    ],
                ),
            ),
            axis=1,  # stick them together side-by-side
        )


@dataclass(frozen=True)
class PipelineSimulator:
    """
    A base class for multi-realization simulations.
    """

    config: PipelineConfig

    @property
    def rume(self) -> RUME:
        """
        The RUME containing the IPM, MM, scope, time_frame, and the fixed params for
        each realization.
        """
        return self.config.rume

    @property
    def num_realizations(self) -> int:
        """
        The number of realzations.
        """
        return self.config.num_realizations

    @property
    def initial_values(self) -> NDArray | None:
        """
        An optional array of initial compartment values of shape (R, N, C) where R is
        the number of realizations, N is the number of nodes, and C is the number of
        compartments. If not specified, the initializer contained in the RUME will be
        used to generate a single initial condition which will be used for all
        realizations.
        """
        return self.config.initial_values

    @property
    def unknown_params(self) -> Mapping[NamePattern, UnknownParam]:
        """
        An array of unknown parameters which will vary across realizations.
        """
        return self.config.unknown_params

    @abstractmethod
    def run(self, rng: np.random.Generator) -> PipelineOutput:
        """
        Run the simulation.
        """


def _initialize_compartments_and_params(
    rume: RUME,
    unknown_params: Mapping[NamePattern, UnknownParam],
    num_realizations: int,
    initial_values: NDArray | None,
    rng: np.random.Generator,
) -> Tuple[NDArray, Mapping[NamePattern, NDArray]]:
    """
    Parameters
    ----------
    rume : RUME
        The RUME.
    unknown_params : Mapping[NamePattern, UnknownParam]
        The unknown parameters. The initial parameter values are based on the prior of
        each UnknownParam.
    num_realizations : int
        The number of realizations.
    initial_values : NDArray | None
        An array of shape (R, N, C) containing the initial compartment values.
        If None then the RUME's intitializer is used to generate a single initial value
        for each compartment.
    rng : np.random.Generator
        The random number generator.

    Returns
    -------
    :
        The initial compartment values and the initial paramter values.
    """
    num_nodes = rume.scope.nodes
    num_compartments = rume.ipm.num_compartments
    current_params = {
        k: np.zeros(shape=(num_realizations, num_nodes), dtype=np.float64)
        for k in unknown_params.keys()
    }
    for i_realization in range(num_realizations):
        for name in current_params.keys():
            prior = unknown_params[name].prior
            if isinstance(prior, Prior):
                current_params[name][i_realization, ...] = prior.sample(
                    size=(num_nodes,), rng=rng
                )
            else:
                current_params[name][i_realization, ...] = prior[i_realization, ...]

    current_compartments = np.zeros(
        shape=(num_realizations, num_nodes, num_compartments), dtype=np.int64
    )
    if initial_values is not None:
        current_compartments = initial_values.copy()
    else:
        params_temp = {**rume.params}
        for i_realization in range(num_realizations):
            for k in unknown_params.keys():
                params_temp[k] = current_params[k][i_realization, ...]
            rume_temp = dataclasses.replace(rume, params=params_temp)
            data = rume_temp.evaluate_params(rng=rng)
            current_compartments[i_realization, ...] = rume.initialize(
                data=data, rng=rng
            )
    return current_compartments, current_params


@dataclass(frozen=True)
class _SimulateRealizationsResult:
    compartments: NDArray
    """
    An array of shape (R, S, N, C) where R is the number of realizations, S is the
    number of tau steps, N is the number of nodes, and C is the number of compartments.
    Contains the compartment values of each realization over time.
    """

    events: NDArray
    """
    An array of shape (R, S, N, E) where R is the number of realizations, S is the
    number of tau steps, N is the number of nodes, and E is the number of events.
    Contains the event values of each realization over time.
    """

    estimated_params: Mapping[NamePattern, NDArray]
    """
    A dictionary containing arrays of shape (R, T, N) where R is the number of
    realizations, T is the number of days, and N is the number of nodes. Each
    array contains the values of the unknown parameters over time.
    """

    predictions: List[NDArray]
    """
    A list of arrays containing the predicted observation corresponding to each
    realization.
    """


def _simulate_realizations(
    rume_template: RUME,
    override_time_frame: TimeFrame,
    num_realizations: int,
    initial_values: NDArray,
    unknown_params: Mapping[NamePattern, UnknownParam],
    param_values: Mapping[NamePattern, NDArray],
    geo: GeoSelection | GeoAggregation,
    time: TimeSelection | TimeAggregation,
    quantity: QuantitySelection | QuantityAggregation,
    rng: np.random.Generator,
) -> _SimulateRealizationsResult:
    """
    A helper function used by ForecastSimulator and ParticleFilterSimulator to propagate
    realizations forward in time.

    Parameters
    ----------
    rume_template : RUME
        A partial RUME which will be filled out with the compartment and parameter
        values of each realization.
    override_time_frame : TimeFrame
        An override of the time_frame of the RUME template.
    num_realizations : int
        The number of realizations.
    initial_values : NDArray
        An array of shape (R, N, C) containing the compartment values of each
        realization.
    unknown_params : Mapping[NamePattern, UnknownParam]
        A dictionary containing the unknown parameters which are allowed to vary across
        realizations. The prior field of each UnknownParam is ingnored in favor of
        param_values.
    param_values : Mapping[NamePattern, NDArray]
        A dictionary containing the current parameter values.
    geo : GeoSelection | GeoAggregation
    time : TimeSelection | TimeAggregation
    quantity : QuantitySelection | QuantityAggregation
    rng : np.random.Generator

    Returns
    -------
    :
        The result of the simulation.
    """
    num_days = override_time_frame.days
    taus = rume_template.num_tau_steps
    num_steps = num_days * taus
    num_nodes = rume_template.scope.nodes
    num_compartments = rume_template.ipm.num_compartments
    num_events = rume_template.ipm.num_events

    compartments = np.zeros(
        shape=(num_realizations, num_steps, num_nodes, num_compartments), dtype=np.int64
    )
    events = np.zeros(
        shape=(num_realizations, num_steps, num_nodes, num_events), dtype=np.int64
    )
    estimated_params = {
        k: np.zeros(shape=(num_realizations, num_days, num_nodes), dtype=np.float64)
        for k in unknown_params.keys()
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
class ForecastSimulator(PipelineSimulator):
    """
    PipelineSimulator for running a multi-realization forecast of a RUME and any
    unknown parameters.

    Parameters
    ----------
    config : PipelineConfig
        Contains the rume, number of realizations, initial compartment values, and
        dictionary of unknown parameters.
    """

    config: PipelineConfig

    def run(self, rng) -> PipelineOutput:
        """
        Runs the multi-realization forecast.

        Parameters
        ----------
        rng : np.random.Generator
            The random number generator.

        Returns
        -------
        :
            The output of the forecast.
        """
        rume = self.rume
        num_realizations = self.num_realizations
        initial_values = self.initial_values
        unknown_params = self.unknown_params
        rng = rng

        # Precompute ADRIO values.
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
        current_compartments, current_params = _initialize_compartments_and_params(
            rume, unknown_params, num_realizations, initial_values, rng
        )
        initial = current_compartments

        result = _simulate_realizations(
            rume_template=rume,
            override_time_frame=rume.time_frame,
            num_realizations=num_realizations,
            initial_values=current_compartments,
            unknown_params=unknown_params,
            param_values=current_params,
            geo=rume.scope.select.all(),
            time=rume.time_frame.select.all(),
            quantity=rume.ipm.select.all(),
            rng=rng,
        )

        return PipelineOutput(
            simulator=self,
            final_compartments=result.compartments[:, -1, ...],
            final_params={
                k: result.estimated_params[k][:, -1, ...] for k in unknown_params.keys()
            },
            compartments=result.compartments,
            events=result.events,
            initial=initial,
            estimated_params=result.estimated_params,
        )


@dataclass(frozen=True)
class ParticleFilterOutput(PipelineOutput):
    """
    Output object for the particle filter which contains additional output and
    diagnostic information.
    """

    simulator: "ParticleFilterSimulator"

    posterior_values: NDArray
    """
    The posterior estimate of the observed data. The first dimension of the array is the
    number of realizations, the second dimension is the number of observations. The
    remaining dimensions depend on the shape of the observed data. These values are
    constructed from resampling the observed data in the same way the compartment and
    parameter values are resampled.
    """

    effective_sample_size: NDArray
    """
    The effective sample size for each observation and each node.
    """


@dataclass(frozen=True)
class ModelLink:
    """
    Contains the information needed to subselect, group, and aggregate model output in
    order to compare it to observed data.
    """

    geo: GeoSelection | GeoAggregation
    time: TimeSelection | TimeAggregation
    quantity: QuantitySelection | QuantityAggregation


@dataclass(frozen=True)
class Observations(Generic[ValueT]):
    """
    The observed data used to estimate the state and parameters of a simulation.
    """

    source: ADRIO[DateValueType, ValueT] | NDArray[DateValueType]
    """
    The source of the observations which results in an (A, N) array where A is a time
    axis of unknown length and N is the number of nodes. The entries of the array are
    date-value pairs.
    """

    model_link: ModelLink
    """
    Contains the information needed to generate a predicted observation from a
    single realization of a simulation.
    """

    likelihood: Likelihood
    """
    Contains a likelihood function used to compare a predicted observation to observed
    data.
    """

    def _get_observations_array(self, context: Context) -> NDArray[DateValueType]:
        """
        Get the arbitrary by node structured array of date value pairs.
        """
        if isinstance(self.source, ADRIO):
            inspect = self.source.with_context_internal(context).inspect()
            return inspect.result
        else:
            return self.source


@dataclass(frozen=True)
class ParticleFilterSimulator(PipelineSimulator):
    """
    A PipelineSimulator for using a particle filter to estimate the state and parameters
    of a system based on observed data.

    Parameters
    ----------
    config : PipelineConfig
        The RUME, number of realization, initial (prior) compartment values, and unknown
        parameters to estimate.
    observations : Observations
        The observations used to estimate the compartment and parameter values.
    """

    config: PipelineConfig
    observations: Observations

    def run(self, rng) -> ParticleFilterOutput:
        """
        Run the particle filter simulation.

        Parameters
        ----------
        rng : np.random.Generator
            The random number generator.

        Returns
        -------
        :
            The output of the particle filter.
        """
        rume = self.rume
        num_realizations = self.num_realizations
        initial_values = self.initial_values
        unknown_params = self.unknown_params
        observations = self.observations
        rng = rng

        # Precompute ADRIO values.
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
        num_days = rume.time_frame.days
        num_steps = num_days * rume.num_tau_steps
        num_nodes = rume.scope.nodes
        num_compartments = rume.ipm.num_compartments
        num_events = rume.ipm.num_events

        # Allocate return values.
        initial = np.zeros(
            shape=(num_realizations, num_nodes, num_compartments), dtype=np.int64
        )
        compartments = np.zeros(
            shape=(num_realizations, num_steps, num_nodes, num_compartments),
            dtype=np.int64,
        )
        events = np.zeros(
            shape=(num_realizations, num_steps, num_nodes, num_events), dtype=np.int64
        )
        estimated_params = {
            name: np.zeros(
                shape=(num_realizations, num_days, num_nodes), dtype=np.float64
            )
            for name in unknown_params.keys()
        }

        # Initialize the initial compartment and parameter values. Note, the values
        # within current_compartment_values will be modified in place!
        current_compartments, current_params = _initialize_compartments_and_params(
            rume=rume,
            unknown_params=unknown_params,
            num_realizations=num_realizations,
            initial_values=initial_values,
            rng=rng,
        )

        initial = current_compartments.copy()

        # Determine the number of observations and their associated time frames.
        time_frames, labels = self._observation_time_frames(
            rume, observations.model_link.time.grouping
        )

        context = Context.of(
            scope=rume.scope, time_frame=rume.time_frame, ipm=rume.ipm, rng=rng
        )
        observations_array = observations._get_observations_array(context)

        posterior_values = []

        effective_sample_size = np.zeros(
            shape=(len(time_frames), num_nodes), dtype=np.float64
        )

        day_idx = 0
        step_idx = 0
        for i_observation in range(len(time_frames)):
            time_frame = time_frames[i_observation]

            print(  # noqa: T201
                (
                    f"Observation: {i_observation}, "
                    f"Label: {labels[i_observation]}, "
                    f"Time Frame: {time_frame}"
                )
            )

            result = _simulate_realizations(
                rume_template=rume,
                override_time_frame=time_frame,
                num_realizations=num_realizations,
                initial_values=current_compartments,
                unknown_params=unknown_params,
                param_values=current_params,
                geo=observations.model_link.geo,
                time=observations.model_link.time,
                quantity=observations.model_link.quantity,
                rng=rng,
            )

            day_right = day_idx + time_frame.days
            step_right = step_idx + time_frame.days * rume.num_tau_steps
            compartments[:, step_idx:step_right, ...] = result.compartments
            events[:, step_idx:step_right, ...] = result.events

            current_params = {}
            for name in unknown_params.keys():
                estimated_params[name][:, day_idx:day_right, ...] = (
                    result.estimated_params[name]
                )
                current_params[name] = result.estimated_params[name][:, -1, ...].copy()

            current_compartments = result.compartments[:, -1, ...].copy()
            current_predictions = np.stack(result.predictions).reshape(
                (num_realizations, num_nodes)
            )

            start = np.datetime64(time_frame.start_date.isoformat())
            end = np.datetime64(time_frame.end_date.isoformat())
            time_frame_mask = (observations_array["date"][:, 0] >= start) & (
                observations_array["date"][:, 0] <= end
            )
            current_observation = observations_array[time_frame_mask, :].reshape((-1,))

            # posterior_value will be modified in-place.
            posterior_value = current_predictions.copy()
            prior_compartment_values = current_compartments.copy()

            # Hard code localization
            for i_node in range(num_nodes):
                observation_is_missing = (
                    current_observation.size == 0
                ) or np.ma.is_masked(current_observation["value"][i_node])

                if observation_is_missing:
                    effective_sample_size[i_observation, i_node] = 1.0
                else:
                    log_likelihoods = observations.likelihood.compute_log(
                        current_observation["value"][i_node],
                        current_predictions[:, i_node],
                    )
                    weights = self._normalize_log_weights(log_likelihoods)

                    resampled_idx = self._systematic_resampling(weights, rng)

                    orig_idx = np.arange(0, num_realizations)
                    current_compartments[orig_idx, i_node, ...] = (
                        prior_compartment_values[resampled_idx, i_node, ...]
                    )
                    for name in current_params.keys():
                        current_params[name][orig_idx, i_node] = current_params[name][
                            resampled_idx, i_node
                        ]

                    posterior_value[orig_idx, i_node, ...] = posterior_value[
                        resampled_idx, i_node, ...
                    ]
                    effective_sample_size[i_observation, i_node] = 1 / np.sum(
                        weights**2
                    )

            posterior_values.append(posterior_value)

            day_idx = day_right
            step_idx = step_right

        return ParticleFilterOutput(
            simulator=self,
            final_compartments=current_compartments,
            final_params=current_params,
            compartments=compartments,
            events=events,
            initial=initial,
            posterior_values=np.array(posterior_values),
            effective_sample_size=effective_sample_size,
            estimated_params=estimated_params,
        )

    @staticmethod
    def _observation_time_frames(
        rume: RUME, time_grouping: TimeGrouping | None
    ) -> Tuple[Sequence[TimeFrame], NDArray[GroupKeyType]]:
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
        if time_grouping is None:
            labels = np.broadcast_to(np.array(["*"]), ticks.shape)
        else:
            labels = time_grouping.map(dim, ticks, dates)

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

    @staticmethod
    def _normalize_log_weights(log_weights):
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
    def _systematic_resampling(weights: NDArray, rng: np.random.Generator):
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


@dataclass(frozen=True)
class EnsembleKalmanFilterOutput(PipelineOutput):
    """
    Output object for the ensemble Kalman filter which contains additional output and
    diagnostic information.
    """

    simulator: "EnsembleKalmanFilterSimulator"

    posterior_values: NDArray
    """
    The posterior estimate of the observed data. The first dimension of the array is the
    number of realizations, the second dimension is the number of observations. The
    remaining dimensions depend on the shape of the observed data. These values are
    constructed updating the observed data in the same way the compartment and
    parameter values are updated.
    """


@dataclass(frozen=True)
class EnsembleKalmanFilterSimulator(PipelineSimulator):
    """
    A PipelineSimulator for using an ensemble Kalman filter to estimate the state and
    parameters of a system based on observed data.

    Parameters
    ----------
    config : PipelineConfig
        The RUME, number of realization, initial (prior) compartment values, and unknown
        parameters to estimate.
    observations : Observations
        The observations used to estimate the compartment and parameter values.
    """

    config: PipelineConfig
    observations: Observations

    def run(self, rng) -> EnsembleKalmanFilterOutput:
        rume = self.rume
        num_realizations = self.num_realizations
        initial_values = self.initial_values
        unknown_params = self.unknown_params
        observations = self.observations
        rng = rng

        if not isinstance(observations.likelihood, Gaussian):
            msg = (
                "The ensemble Kalman filter only supports a Gaussian"
                "likelihood for observational data."
            )
            raise ValueError(msg)

        # Precompute ADRIO values.
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
        num_days = rume.time_frame.days
        num_steps = num_days * rume.num_tau_steps
        num_nodes = rume.scope.nodes
        num_compartments = rume.ipm.num_compartments
        num_events = rume.ipm.num_events

        # Allocate return values.
        initial = np.zeros(
            shape=(num_realizations, num_nodes, num_compartments), dtype=np.int64
        )
        compartments = np.zeros(
            shape=(num_realizations, num_steps, num_nodes, num_compartments),
            dtype=np.int64,
        )
        events = np.zeros(
            shape=(num_realizations, num_steps, num_nodes, num_events), dtype=np.int64
        )
        estimated_params = {
            name: np.zeros(
                shape=(num_realizations, num_days, num_nodes), dtype=np.float64
            )
            for name in unknown_params.keys()
        }

        # Initialize the initial compartment and parameter values. Note, the values
        # within current_compartment_values will be modified in place!
        current_compartments, current_params = _initialize_compartments_and_params(
            rume=rume,
            unknown_params=unknown_params,
            num_realizations=num_realizations,
            initial_values=initial_values,
            rng=rng,
        )

        initial = current_compartments.copy()

        # Determine the number of observations and their associated time frames.
        time_frames, labels = ParticleFilterSimulator._observation_time_frames(
            rume, observations.model_link.time.grouping
        )

        context = Context.of(
            scope=rume.scope, time_frame=rume.time_frame, ipm=rume.ipm, rng=rng
        )
        observations_array = observations._get_observations_array(context)

        posterior_values = []

        day_idx = 0
        step_idx = 0
        for i_observation in range(len(time_frames)):
            time_frame = time_frames[i_observation]

            print(  # noqa: T201
                (
                    f"Observation: {i_observation}, "
                    f"Label: {labels[i_observation]}, "
                    f"Time Frame: {time_frame}"
                )
            )

            result = _simulate_realizations(
                rume_template=rume,
                override_time_frame=time_frame,
                num_realizations=num_realizations,
                initial_values=current_compartments,
                unknown_params=unknown_params,
                param_values=current_params,
                geo=observations.model_link.geo,
                time=observations.model_link.time,
                quantity=observations.model_link.quantity,
                rng=rng,
            )

            day_right = day_idx + time_frame.days
            step_right = step_idx + time_frame.days * rume.num_tau_steps
            compartments[:, step_idx:step_right, ...] = result.compartments
            events[:, step_idx:step_right, ...] = result.events

            current_params = {}
            for name in unknown_params.keys():
                estimated_params[name][:, day_idx:day_right, ...] = (
                    result.estimated_params[name]
                )
                current_params[name] = result.estimated_params[name][:, -1, ...].copy()

            current_compartments = result.compartments[:, -1, ...].copy()
            current_predictions = np.stack(result.predictions).reshape(
                (num_realizations, num_nodes)
            )

            start = np.datetime64(time_frame.start_date.isoformat())
            end = np.datetime64(time_frame.end_date.isoformat())
            time_frame_mask = (observations_array["date"][:, 0] >= start) & (
                observations_array["date"][:, 0] <= end
            )
            current_observation = observations_array[time_frame_mask, :].reshape((-1,))

            # posterior_value will be modified in-place.
            posterior_value = current_predictions.copy()

            observation_cov = observations.likelihood.variance

            # Hard code localization
            for i_node in range(num_nodes):
                observation_is_missing = (
                    current_observation.size == 0
                ) or np.ma.is_masked(current_observation["value"][i_node])

                if not observation_is_missing:
                    prior_compartments_mean = np.mean(
                        current_compartments[:, i_node, :], axis=0
                    )
                    prior_params_mean = {
                        k: np.mean(current_params[k][:, i_node])
                        for k in unknown_params.keys()
                    }

                    # The double use of the term perturbation is unfortunate.
                    compartments_perturbation = (
                        # Relies on numpy broadcasting rules, i.e. left padding shape.
                        current_compartments[:, i_node, :] - prior_compartments_mean
                    )
                    params_perturbation = {
                        k: current_params[k][:, i_node] - prior_params_mean[k]
                        for k in unknown_params.keys()
                    }
                    prediction_perturbation = (
                        current_predictions[:, i_node]
                        - current_predictions[:, i_node].mean()
                    )

                    prediction_cov = np.var(current_predictions[:, i_node])
                    residual_cov_inverse = 1 / (prediction_cov + observation_cov)

                    # np.matmul behaves nicely with 1-D arrays.
                    # fmt: off
                    kalman_gain_compartments = 1 / (num_realizations - 1) * np.matmul(compartments_perturbation.T, prediction_perturbation) * residual_cov_inverse  # noqa: E501
                    kalman_gain_params = {
                        k: 1 / (num_realizations - 1) * np.matmul(params_perturbation[k], prediction_perturbation) * residual_cov_inverse for k in unknown_params.keys() # noqa: E501
                    }
                    kalman_gain_prediction = 1 / (num_realizations - 1) * np.matmul(prediction_perturbation, prediction_perturbation) * residual_cov_inverse # noqa: E501
                    # fmt: on

                    observation = current_observation["value"][i_node]
                    for i_realization in range(num_realizations):
                        prediction = current_predictions[i_realization, i_node]
                        perturbation = rng.normal(loc=0, scale=np.sqrt(observation_cov))
                        innovation = observation - (prediction + perturbation)
                        # fmt: off
                        current_compartments[i_realization, i_node, :] += np.rint(kalman_gain_compartments * innovation).astype(np.int64) # noqa: E501
                        for name in unknown_params.keys():
                            current_params[name][i_realization, i_node] += kalman_gain_params[name] * innovation  # noqa: E501

                        posterior_value[i_realization, i_node] += kalman_gain_prediction * innovation  # noqa: E501
                        # fmt: on
                    current_compartments[:, i_node, :] = np.clip(
                        current_compartments[:, i_node, :], a_min=0, a_max=None
                    )

                    posterior_value[:, i_node] = np.clip(
                        posterior_value[:, i_node], a_min=0, a_max=None
                    )

            posterior_values.append(posterior_value)

            day_idx = day_right
            step_idx = step_right

        return EnsembleKalmanFilterOutput(
            simulator=self,
            final_compartments=current_compartments,
            final_params=current_params,
            compartments=compartments,
            events=events,
            initial=initial,
            posterior_values=np.array(posterior_values),
            estimated_params=estimated_params,
        )
