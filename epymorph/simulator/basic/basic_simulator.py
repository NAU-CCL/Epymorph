"""Implements one epymorph simulation algorithm: the basic simulator."""

from typing import Callable, Mapping

import numpy as np

from epymorph.data_type import SimDType
from epymorph.error import (
    AttributeException,
    CompilationException,
    InitException,
    IpmSimException,
    MmSimException,
    SimValidationException,
    ValidationException,
    error_gate,
)
from epymorph.event import EventBus, OnStart, OnTick
from epymorph.geography.scope import GeoScope
from epymorph.rume import GEO_LABELS, Rume
from epymorph.simulation import ParamValue, simulation_clock
from epymorph.simulator.basic.ipm_exec import IpmExecutor
from epymorph.simulator.basic.mm_exec import MovementExecutor
from epymorph.simulator.basic.output import Output
from epymorph.simulator.world_list import ListWorld
from epymorph.time import TimeFrame

_events = EventBus()


class BasicSimulator:
    """
    A simulator for running singular simulation passes and producing time-series output.
    The most basic simulator!
    """

    rume: Rume[GeoScope]
    ipm_exec: IpmExecutor
    mm_exec: MovementExecutor

    def __init__(self, rume: Rume[GeoScope]):
        self.rume = rume

    def run(
        self,
        /,
        params: Mapping[str, ParamValue] | None = None,
        time_frame: TimeFrame | None = None,
        rng_factory: Callable[[], np.random.Generator] | None = None,
    ) -> Output:
        """Run a RUME with the given overrides."""

        rume = self.rume
        if time_frame is not None:
            rume = rume.with_time_frame(time_frame)

        dim = rume.dim
        rng = (rng_factory or np.random.default_rng)()

        with error_gate(
            "evaluating simulation attributes",
            ValidationException,
            CompilationException,
        ):
            try:
                data = rume.evaluate_params(override_params=params, rng=rng)
            except AttributeException as e:
                msg = f"RUME attribute requirements were not met. See errors:\n- {e}"
                raise SimValidationException(msg) from None
            except ExceptionGroup as e:
                msg = "RUME attribute requirements were not met. See errors:" + "".join(
                    f"\n- {e}" for e in e.exceptions
                )
                raise SimValidationException(msg) from None

        with error_gate("initializing the simulation", InitException):
            initial_values = rume.initialize(data, rng)
            world = ListWorld.from_initials(initial_values)

        with error_gate("compiling the simulation", CompilationException):
            ipm_exec = IpmExecutor(rume, world, data, rng)
            movement_exec = MovementExecutor(rume, world, data, rng)

        _events.on_start.publish(OnStart(self.__class__.__name__, dim, rume.time_frame))

        S, N, C, E = dim.ticks, dim.nodes, dim.compartments, dim.events
        visit_compartments = np.zeros((S, N, C), dtype=SimDType)
        visit_events = np.zeros((S, N, E), dtype=SimDType)
        home_compartments = np.zeros((S, N, C), dtype=SimDType)
        home_events = np.zeros((S, N, E), dtype=SimDType)

        # Run the simulation!
        for tick in simulation_clock(dim):
            t = tick.sim_index

            # First do movement
            with error_gate("executing movement", MmSimException, AttributeException):
                movement_exec.apply(tick)

            # Then do IPM
            with error_gate("executing the IPM", IpmSimException, AttributeException):
                vcs, ves, hcs, hes = ipm_exec.apply(tick)
                visit_compartments[t] = vcs
                visit_events[t] = ves
                home_compartments[t] = hcs
                home_events[t] = hes

            _events.on_tick.publish(OnTick(t, (t + 1) / dim.ticks, dim))

        _events.on_finish.publish(None)

        # Assemble output.

        # This should always match because `evaluate_params` includes a default.
        if (labels_value := data.resolve(GEO_LABELS.key, GEO_LABELS.value)) is not None:
            geo_labels = labels_value.tolist()
        else:
            geo_labels = rume.scope.labels.tolist()

        return Output(
            dim=dim,
            scope=rume.scope,
            geo_labels=geo_labels,
            time_frame=rume.time_frame,
            ipm=rume.ipm,
            initial=initial_values,
            visit_compartments=visit_compartments,
            visit_events=visit_events,
            home_compartments=home_compartments,
            home_events=home_events,
        )
