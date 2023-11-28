"""
The most fundamental epymorph simulation type:
run a single simulation from start to finish with a static set of parameters.
"""
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from epymorph.compartment_model import CompartmentModel
from epymorph.engine.context import RumeConfig, RumeContext
from epymorph.engine.ipm_exec import StandardIpmExecutor
from epymorph.engine.mm_exec import StandardMovementExecutor
from epymorph.engine.world_list import ListWorld
from epymorph.error import (AttributeException, CompilationException,
                            InitException, IpmSimException, MmSimException,
                            ValidationException, error_gate)
from epymorph.geo.geo import Geo
from epymorph.initializer import DEFAULT_INITIALIZER, Initializer
from epymorph.movement.movement_model import MovementModel
from epymorph.movement.parser import MovementSpec
from epymorph.params import Params
from epymorph.simulation import (OnStart, SimDimensions, SimDType, SimTick,
                                 SimulationEvents, TimeFrame)
from epymorph.util import Event


@dataclass
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

    prevalence: NDArray[SimDType] = field(init=False)
    """
    Prevalence data by timestep, population, and compartment.
    Array of shape (T,N,C) where T is the number of ticks in the simulation,
    N is the number of populations, and C is the number of compartments.
    """

    incidence: NDArray[SimDType] = field(init=False)
    """
    Incidence data by timestep, population, and event.
    Array of shape (T,N,E) where T is the number of ticks in the simulation,
    N is the number of populations, and E is the number of events.
    """

    def __post_init__(self):
        T, N, C, E = self.dim.TNCE
        self.prevalence = np.zeros((T, N, C), dtype=SimDType)
        self.incidence = np.zeros((T, N, E), dtype=SimDType)


class StandardSimulation(SimulationEvents):
    """Runs singular simulation passes, producing time-series output."""

    _config: RumeConfig
    on_tick: Event[SimTick]  # this class supports on_tick; so narrow the type def

    def __init__(self,
                 geo: Geo,
                 ipm: CompartmentModel,
                 mm: MovementModel | MovementSpec,
                 params: Params,
                 time_frame: TimeFrame,
                 initializer: Initializer | None = None,
                 rng: Callable[[], np.random.Generator] | None = None):
        if initializer is None:
            initializer = DEFAULT_INITIALIZER
        if rng is None:
            rng = np.random.default_rng

        self._config = RumeConfig(geo, ipm, mm, params, time_frame, initializer, rng)

        # events
        self.on_start = Event()
        self.on_tick = Event()
        self.on_end = Event()

    def validate(self) -> None:
        """Validate the simulation."""
        with error_gate("validating the simulation", ValidationException, CompilationException):
            ctx = RumeContext.from_config(self._config)
            # ctx.validate_geo() # validate only the required geo parameters?
            # ctx.validate_mm()
            ctx.validate_ipm()
            # ctx.validate_init()

    def run(self) -> Output:
        """
        Run the simulation. It is safe to call this multiple times
        to run multiple independent simulations with the same configuraiton.
        """
        with error_gate("compiling the simulation", CompilationException):
            ctx = RumeContext.from_config(self._config)
            ipm_exec = StandardIpmExecutor(ctx)
            movement_exec = StandardMovementExecutor(ctx)

        with error_gate("initializing the simulation", InitException):
            ini = ctx.initialize()
            world = ListWorld.from_initials(ini)
            out = Output(ctx.dim, ctx.geo.labels.tolist(),
                         ctx.ipm.compartment_names, ctx.ipm.event_names, ini)

        self.on_start.publish(OnStart(dim=ctx.dim, time_frame=ctx.time_frame))

        for tick in ctx.clock():
            # First do movement
            with error_gate("executing_the_movement_model", MmSimException, AttributeException):
                movement_exec.apply(world, tick)

            # Then do IPM
            with error_gate("executing the IPM", IpmSimException, AttributeException):
                tick_events, tick_prevalence = ipm_exec.apply(world, tick)
                out.incidence[tick.index] = tick_events
                out.prevalence[tick.index] = tick_prevalence

            t = tick.index
            self.on_tick.publish(SimTick(t, (t + 1) / ctx.dim.ticks))

        self.on_end.publish(None)
        return out
