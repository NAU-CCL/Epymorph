from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

import epymorph.error as error
from epymorph.engine.context import (ExecutionConfig, ExecutionContext,
                                     build_execution_context)
from epymorph.engine.ipm_exec import StandardIpmExecutor
from epymorph.engine.mm_exec import StandardMovementExecutor
from epymorph.engine.world_list import ListWorld
from epymorph.simulation import (OnStart, SimDimensions, SimDType, SimTick,
                                 SimulationEvents)
from epymorph.util import Event


class Output:
    """
    The output of a simulation run, including prevalence for all populations and all IPM compartments
    and incidence for all populations and all IPM events.
    """

    dim: SimDimensions
    geo_labels: Sequence[str]
    compartment_labels: Sequence[str]
    event_labels: Sequence[str]

    initial: NDArray[SimDType]
    """
    Initial prevalence data by population and compartment.
    Array of shape (N, C) where N is the number of populations, and C is the number of compartments
    """

    prevalence: NDArray[SimDType]
    """
    Prevalence data by timestep, population, and compartment.
    Array of shape (T,N,C) where T is the number of ticks in the simulation,
    N is the number of populations, and C is the number of compartments.
    """

    incidence: NDArray[SimDType]
    """
    Incidence data by timestep, population, and event.
    Array of shape (T,N,E) where T is the number of ticks in the simulation,
    N is the number of populations, and E is the number of events.
    """

    def __init__(self,
                 dim: SimDimensions,
                 geo_labels: Sequence[str],
                 compartment_labels: Sequence[str],
                 event_labels: Sequence[str],
                 initial: NDArray[SimDType]):
        self.dim = dim
        self.geo_labels = geo_labels
        self.compartment_labels = compartment_labels
        self.event_labels = event_labels
        self.initial = initial
        T, N, C, E = dim.TNCE
        self.prevalence = np.zeros((T, N, C), dtype=SimDType)
        self.incidence = np.zeros((T, N, E), dtype=SimDType)


class StandardSimulation(SimulationEvents):
    """Runs a single simulation pass, producing time-series output."""

    _config: ExecutionConfig
    on_tick: Event[SimTick]  # this class supports on_tick; so narrow the type def

    def __init__(self, config: ExecutionConfig):
        self._config = config
        # events
        self.on_start = Event()
        self.on_tick = Event()
        self.on_end = Event()

    def _make_context(self) -> ExecutionContext:
        return build_execution_context(self._config)

    def validate(self) -> None:
        try:
            ctx = self._make_context()
            ctx.validate_ipm()
            # ctx.validate_mm()
            # ctx.validate_init()
        except error.ValidationException as e:
            raise e
        except Exception as e:
            msg = "Unknown error validating the simulation."
            raise error.ValidationException(msg) from e

    def run(self) -> Output:
        try:
            ctx = self._make_context()
            ini = ctx.initialize()
            world = ListWorld.from_initials(ini)
            ipm_exec = StandardIpmExecutor(ctx)
            movement_exec = StandardMovementExecutor(ctx)
        except error.InitException as e:
            raise e
        except Exception as e:
            msg = "Unknown error compiling simulation."
            raise error.SimCompileException(msg) from e

        self.on_start.publish(OnStart(dim=ctx.dim, time_frame=ctx.time_frame))

        out = Output(ctx.dim, ctx.geo.labels.tolist(),
                     ctx.ipm.compartment_names, ctx.ipm.event_names, ini)

        for tick in ctx.clock():
            # First do movement
            try:
                movement_exec.apply(world, tick)
            except error.MmSimException as e:
                raise e
            except Exception as e:
                msg = "Unknown error simulating the movement model."
                raise error.MmSimException(msg) from e

            # Then do IPM
            try:
                tick_events, tick_prevalence = ipm_exec.apply(world, tick)
                out.incidence[tick.index] = tick_events
                out.prevalence[tick.index] = tick_prevalence
            except error.IpmSimException as e:
                raise e
            except Exception as e:
                msg = "Unknown error simulating the IPM."
                raise error.IpmSimException(msg) from e

            t = tick.index
            self.on_tick.publish(SimTick(t, (t + 1) / ctx.dim.ticks))

        self.on_end.publish(None)
        return out
