"""
Implements classes for representing observed data along with a base class for filtering
algorithms for state/parameter estimation.
"""

import dataclasses
import datetime
from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, Mapping, Sequence, TypeVar

import numpy as np
from numpy.typing import NDArray

from epymorph.adrio.adrio import ADRIO, ValueT
from epymorph.attribute import NamePattern
from epymorph.compartment_model import (
    QuantityAggregation,
    QuantitySelection,
)
from epymorph.data_type import SimDType
from epymorph.event import EventBus, OnPipelineStart
from epymorph.forecasting.likelihood import Likelihood
from epymorph.forecasting.pipeline import (
    PipelineConfig,
    PipelineOutput,
    PipelineSimulator,
    _initialize_compartments_and_params,
    _simulate_realizations,
)
from epymorph.geography.scope import GeoAggregation, GeoSelection
from epymorph.rume import RUME
from epymorph.simulation import Context
from epymorph.time import (
    Dim,
    GroupKeyType,
    TimeAggregation,
    TimeFrame,
    TimeGrouping,
    TimeSelection,
)
from epymorph.util import DateValueType

_events = EventBus()


@dataclass(frozen=True)
class ModelLink:
    """
    Contains the information needed to subselect, group, and aggregate model output in
    order to compare it to observed data.

    Parameters
    ----------
    geo :
        The strategy for processing the geo-axis of a single realization.
    time :
        The strategy for processing the time-axis of a single realization.
    quantity :
        The strategy for processing the quantity-axis of a single realization.
    """

    geo: GeoSelection | GeoAggregation
    """The strategy for processing the geo-axis of a single realization."""

    time: TimeSelection | TimeAggregation
    """The strategy for processing the time-axis of a single realization."""

    quantity: QuantitySelection | QuantityAggregation
    """The strategy for processing the quantity-axis of a single realization."""


@dataclass(frozen=True)
class Observations(Generic[ValueT]):
    """
    The observed data used to estimate the state and parameters of a simulation.

    Parameters
    ----------
    source :
        The source of the observations which results in an array of shape (A, N) or
        (A, N, Q) array where A is a time axis of unknown length, N is the number of
        nodes, and Q are quantities. The entries of the array are date-value pairs.
    model_link :
        Contains the information needed to generate a predicted observation from a
        single realization of a simulation.
    likelihood :
        Contains a likelihood function used to compare a predicted observation to
        observed data.
    """

    source: ADRIO[DateValueType, ValueT] | NDArray[DateValueType]
    """
    The source of the observations which results in an  array of observed values.
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

        Parameters
        ----------
        context :
            The context for evaluating an ADRIO source of the observations.
        """
        if isinstance(self.source, ADRIO):
            inspect = self.source.with_context_internal(context).inspect()
            return inspect.result
        else:
            return self.source


_FilterContextT = TypeVar("_FilterContextT")


@dataclass(frozen=True)
class _FilterUpdateResult(Generic[_FilterContextT]):
    """
    Result of the update step of a filter algorithm.
    """

    current_compartments: NDArray
    current_params: Mapping[NamePattern, NDArray]
    posterior_values: np.ndarray
    filter_context: _FilterContextT


@dataclass(frozen=True)
class _RunFilterResult(Generic[_FilterContextT]):
    """
    The raw output of a FilterSimulator. Subclasses should process this raw output into
    an appropriate FilterOutput.
    """

    final_compartments: NDArray[SimDType]
    final_params: Mapping[NamePattern, NDArray[np.float64]]
    compartments: NDArray[SimDType]
    events: NDArray[SimDType]
    initial: NDArray[SimDType]
    estimated_params: Mapping[NamePattern, NDArray[np.float64]]
    posterior_values: NDArray
    filter_context: _FilterContextT


@dataclass(frozen=True)
class FilterOutput(PipelineOutput):
    """
    Output for a FilterSimulator.
    """

    simulator: "FilterSimulator"
    posterior_values: NDArray[np.float64]


@dataclass(frozen=True)
class FilterSimulator(PipelineSimulator, Generic[_FilterContextT]):
    """
    A abstract class for running a filter algorithm. It is generic in the filter context
    which contains diagnostic information particular to each filter algorithm.
    """

    config: PipelineConfig
    """The configuration for the simulation."""

    observations: Observations
    """The observations."""

    @abstractmethod
    def _initialize_filter_context(
        self,
    ) -> _FilterContextT:
        """Initialize the filter context which is passed to the update step."""

    @abstractmethod
    def run(self, rng: np.random.Generator) -> FilterOutput:
        """Run the filter."""

    @abstractmethod
    def _update(
        self,
        current_observation: NDArray,
        current_predictions: NDArray,
        prior_compartments: NDArray,
        prior_params: Mapping[NamePattern, NDArray],
        filter_context: _FilterContextT,
        rng: np.random.Generator,
    ) -> _FilterUpdateResult:
        """Run the update step of the filter."""

    def _run_filter(self, rng: np.random.Generator) -> _RunFilterResult:
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

        # Dimension of the system.
        num_steps = rume.time_frame.days * rume.num_tau_steps

        # Allocate return values.
        initial = np.zeros(
            shape=(self.num_realizations, rume.scope.nodes, rume.ipm.num_compartments),
            dtype=np.int64,
        )
        compartments = np.zeros(
            shape=(
                self.num_realizations,
                num_steps,
                rume.scope.nodes,
                rume.ipm.num_compartments,
            ),
            dtype=np.int64,
        )
        events = np.zeros(
            shape=(
                self.num_realizations,
                num_steps,
                rume.scope.nodes,
                rume.ipm.num_events,
            ),
            dtype=np.int64,
        )
        estimated_params = {
            name: np.zeros(
                shape=(self.num_realizations, rume.time_frame.days, rume.scope.nodes),
                dtype=np.float64,
            )
            for name in self.unknown_params.keys()
        }

        # Initialize the initial compartment and parameter values. Note, the values
        # within current_compartment_values will be modified in place!
        current_compartments, current_params = _initialize_compartments_and_params(
            rume=rume,
            unknown_params=self.unknown_params,
            num_realizations=self.num_realizations,
            initial_values=self.initial_values,
            rng=rng,
        )

        initial = current_compartments.copy()

        # Determine the number of observations and their associated time frames.
        time_frames, labels = self._observation_time_frames(
            rume, self.observations.model_link.time.grouping
        )

        context = Context.of(
            scope=rume.scope, time_frame=rume.time_frame, ipm=rume.ipm, rng=rng
        )
        observations_array = self.observations._get_observations_array(context)
        observations_array = observations_array.reshape(
            (observations_array.shape[0], observations_array.shape[1], -1)
        )

        name = self.__class__.__name__
        _events.on_pipeline_start.publish(
            OnPipelineStart(
                name,
                rume,
                self.num_realizations,
                self.num_realizations * len(time_frames),
            )
        )

        posterior_values = []

        filter_context = self._initialize_filter_context()

        day_idx = 0
        step_idx = 0
        for i_observation in range(len(time_frames)):
            time_frame = time_frames[i_observation]

            result = _simulate_realizations(
                rume_template=rume,
                override_time_frame=time_frame,
                num_realizations=self.num_realizations,
                initial_values=current_compartments,
                unknown_params=self.unknown_params,
                param_values=current_params,
                geo=self.observations.model_link.geo,
                time=self.observations.model_link.time,
                quantity=self.observations.model_link.quantity,
                rng=rng,
            )

            day_right = day_idx + time_frame.days
            step_right = step_idx + time_frame.days * rume.num_tau_steps
            compartments[:, step_idx:step_right, ...] = result.compartments
            events[:, step_idx:step_right, ...] = result.events

            current_params = {}
            for name in estimated_params.keys():
                estimated_params[name][:, day_idx:day_right, ...] = (
                    result.estimated_params[name]
                )
                current_params[name] = result.estimated_params[name][:, -1, ...].copy()

            current_compartments = result.compartments[:, -1, ...].copy()
            current_predictions = np.stack(result.predictions).reshape(
                (self.num_realizations, rume.scope.nodes, -1)
            )

            start = np.datetime64(time_frame.start_date.isoformat())
            end = np.datetime64(time_frame.end_date.isoformat())
            time_frame_mask = (observations_array["date"][:, 0, 0] >= start) & (
                observations_array["date"][:, 0, 0] <= end
            )

            current_observation = observations_array[time_frame_mask, :].reshape(
                (rume.scope.nodes, -1)
            )

            filter_update_result = self._update(
                current_observation=current_observation,
                current_predictions=current_predictions,
                prior_compartments=current_compartments,
                prior_params=current_params,
                filter_context=filter_context,
                rng=rng,
            )

            current_compartments = filter_update_result.current_compartments
            current_params = filter_update_result.current_params
            posterior_value = filter_update_result.posterior_values
            posterior_values.append(posterior_value)
            filter_context = filter_update_result.filter_context

            day_idx = day_right
            step_idx = step_right

        _events.on_pipeline_finish.publish(None)

        return _RunFilterResult(
            final_compartments=current_compartments,
            final_params=current_params,
            compartments=compartments,
            events=events,
            initial=initial,
            posterior_values=np.stack(posterior_values, axis=1),
            estimated_params=estimated_params,
            filter_context=filter_context,
        )

    @staticmethod
    def _observation_time_frames(
        rume: RUME, time_grouping: TimeGrouping | None
    ) -> tuple[Sequence[TimeFrame], NDArray[GroupKeyType]]:
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
