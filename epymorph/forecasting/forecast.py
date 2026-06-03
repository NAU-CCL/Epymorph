import dataclasses
from dataclasses import dataclass

import numpy as np
from typing_extensions import override

from epymorph.adrio.adrio import ADRIO
from epymorph.event import EventBus, OnPipelineStart
from epymorph.forecasting.pipeline import (
    PipelineConfig,
    PipelineOutput,
    PipelineSimulator,
    _initialize_compartments_and_params,
    _simulate_realizations,
)
from epymorph.simulation import Context

_events = EventBus()


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
