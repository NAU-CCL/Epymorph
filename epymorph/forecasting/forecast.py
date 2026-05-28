import dataclasses
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from epymorph.adrio.adrio import ADRIO
from epymorph.attribute import NamePattern
from epymorph.compartment_model import QuantityAggregation, QuantitySelection
from epymorph.data_type import SimDType
from epymorph.event import EventBus, OnPipelineProgress, OnPipelineStart
from epymorph.forecasting.dynamic_params import Prior
from epymorph.forecasting.pipeline import (
    PipelineConfig,
    PipelineOutput,
    PipelineSimulator,
    UnknownParam,
)
from epymorph.geography.scope import GeoAggregation, GeoSelection
from epymorph.initializer import Explicit
from epymorph.rume import RUME
from epymorph.simulation import Context
from epymorph.simulator.basic.basic_simulator import BasicSimulator
from epymorph.time import (
    TimeAggregation,
    TimeFrame,
    TimeSelection,
)
from epymorph.tools.data import munge

_events = EventBus()


def _initialize_compartments_and_params(
    rume: RUME,
    unknown_params: Mapping[NamePattern, UnknownParam],
    num_realizations: int,
    initial_values: NDArray[SimDType] | None,
    rng: np.random.Generator,
) -> tuple[NDArray, Mapping[NamePattern, NDArray]]:
    """
    Parameters
    ----------
    rume :
        The RUME.
    unknown_params :
        The unknown parameters. The initial parameter values are based on the prior of
        each UnknownParam.
    num_realizations :
        The number of realizations.
    initial_values :
        An array of shape (R, N, C) containing the initial compartment values.
        If None then the RUME's intitializer is used to generate a single initial value
        for each compartment.
    rng :
        The random number generator.

    Returns
    -------
    :
        The initial compartment values and the initial paramter values.
    """
    num_compartments = rume.ipm.num_compartments
    current_params = {
        k: np.zeros(shape=(num_realizations, rume.scope.nodes), dtype=np.float64)
        for k in unknown_params.keys()
    }
    for i_realization in range(num_realizations):
        for name in current_params.keys():
            prior = unknown_params[name].prior
            if isinstance(prior, Prior):
                current_params[name][i_realization, ...] = prior.sample(
                    size=(rume.scope.nodes,), rng=rng
                )
            else:
                current_params[name][i_realization, ...] = prior[i_realization, ...]

    current_compartments = np.zeros(
        shape=(num_realizations, rume.scope.nodes, num_compartments), dtype=np.int64
    )
    if initial_values is not None:
        current_compartments = initial_values.copy()
    else:
        params_temp = {**rume.params}
        for i_realization in range(num_realizations):
            for name in current_params.keys():
                params_temp[name] = current_params[name][i_realization, ...]
            rume_temp = dataclasses.replace(rume, params=params_temp)
            data = rume_temp.evaluate_params(rng=rng)
            current_compartments[i_realization, ...] = rume.initialize(
                data=data, rng=rng
            )
    return current_compartments, current_params


@dataclass(frozen=True)
class _SimulateRealizationsResult:
    compartments: NDArray[SimDType]
    """
    An array of shape (R, S, N, C) where R is the number of realizations, S is the
    number of tau steps, N is the number of nodes, and C is the number of compartments.
    Contains the compartment values of each realization over time.
    """

    events: NDArray[SimDType]
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

    predictions: list[NDArray]
    """
    A list of arrays containing the predicted observation corresponding to each
    realization.
    """


def _simulate_realizations(
    rume_template: RUME,
    override_time_frame: TimeFrame,
    num_realizations: int,
    initial_values: NDArray[SimDType],
    unknown_params: Mapping[NamePattern, UnknownParam],
    param_values: Mapping[NamePattern, NDArray[np.float64]],
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
    rume_template :
        A partial RUME which will be filled out with the compartment and parameter
        values of each realization.
    override_time_frame :
        An override of the time_frame of the RUME template.
    num_realizations :
        The number of realizations.
    initial_values :
        An array of shape (R, N, C) containing the compartment values of each
        realization.
    unknown_params :
        A dictionary containing the unknown parameters which are allowed to vary across
        realizations. The prior field of each UnknownParam is ingnored in favor of
        param_values.
    param_values :
        A dictionary containing the current parameter values.
    geo :
        The geo strategy for munging the output.
    time :
        The time strategy for munging the output.
    quantity :
        The quantity strategy for munging the output.
    rng :
        The random number generator.

    Returns
    -------
    :
        The result of the simulation.
    """
    num_steps = override_time_frame.days * rume_template.num_tau_steps

    compartments = np.zeros(
        shape=(
            num_realizations,
            num_steps,
            rume_template.scope.nodes,
            rume_template.ipm.num_compartments,
        ),
        dtype=np.int64,
    )
    events = np.zeros(
        shape=(
            num_realizations,
            num_steps,
            rume_template.scope.nodes,
            rume_template.ipm.num_events,
        ),
        dtype=np.int64,
    )
    estimated_params = {
        k: np.zeros(
            shape=(
                num_realizations,
                override_time_frame.days,
                rume_template.scope.nodes,
            ),
            dtype=np.float64,
        )
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

        for name in estimated_params.keys():
            matching_names = [x for x in data.raw_values if name.match(x)]
            # We know that all the matching names have identical values.
            estimated_params[name][i_realization, ...] = data.raw_values[
                matching_names[0]
            ]

        _events.on_pipeline_progress.publish(OnPipelineProgress(1))

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
    config :
        Contains the rume, number of realizations, initial compartment values, and
        dictionary of unknown parameters.
    """

    config: PipelineConfig
    """The configuration for the simulator."""

    @override
    def run(self, rng: np.random.Generator) -> PipelineOutput:
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

        # Precompute ADRIO values.
        context = Context.of(
            scope=self.rume.scope, time_frame=self.rume.time_frame, rng=rng
        )
        new_params = {}
        for name, param in self.rume.params.items():
            if isinstance(param, ADRIO):
                new_params[name] = param.with_context_internal(context).evaluate()
            else:
                new_params[name] = param
        rume = dataclasses.replace(
            self.rume,
            params=new_params,
        )

        # Initialize the initial compartment and parameter values.
        current_compartments, current_params = _initialize_compartments_and_params(
            rume, self.unknown_params, self.num_realizations, self.initial_values, rng
        )
        initial = current_compartments

        name = self.__class__.__name__
        _events.on_pipeline_start.publish(
            OnPipelineStart(name, rume, self.num_realizations, self.num_realizations)
        )

        result = _simulate_realizations(
            rume_template=rume,
            override_time_frame=rume.time_frame,
            num_realizations=self.num_realizations,
            initial_values=current_compartments,
            unknown_params=self.unknown_params,
            param_values=current_params,
            geo=rume.scope.select.all(),
            time=rume.time_frame.select.all(),
            quantity=rume.ipm.select.all(),
            rng=rng,
        )

        _events.on_pipeline_finish.publish(None)

        return PipelineOutput(
            simulator=self,
            final_compartments=result.compartments[:, -1, ...],
            final_params={
                k: result.estimated_params[k][:, -1, ...]
                for k in result.estimated_params.keys()
            },
            compartments=result.compartments,
            events=result.events,
            initial=initial,
            estimated_params=result.estimated_params,
        )
