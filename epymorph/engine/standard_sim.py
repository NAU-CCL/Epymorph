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
from epymorph.event import (MovementEventsMixin, OnStart, OnTick,
                            SimulationEventsMixin)
from epymorph.geo.geo import Geo
from epymorph.initializer import DEFAULT_INITIALIZER, Initializer
from epymorph.movement.movement_model import MovementModel, validate_mm
from epymorph.movement.parser import MovementSpec
from epymorph.params import ContextParams, Params
from epymorph.simulation import SimDimensions, SimDType, TimeFrame
from epymorph.util import Subscriber


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

    @property
    def incidence_per_day(self) -> NDArray[SimDType]:
        """
        Returns this output's `incidence` from a per-tick value to a per-day value.
        Returns a shape (D,N,E) array, where D is the number of simulation days.
        """
        T, N, _, E = self.dim.TNCE
        taus = self.dim.tau_steps
        return np.sum(
            self.incidence.reshape((T // taus, taus, N, E)),
            axis=1,
            dtype=SimDType
        )

    @property
    def ticks_in_days(self) -> NDArray[np.float64]:
        """
        Create a series with as many values as there are simulation ticks,
        but in the scale of fractional days. That is: the cumulative sum of
        the simulation's tau step lengths across the simulation duration.
        Returns a shape (T,) array, where T is the number of simulation ticks.
        """
        return np.cumsum(np.tile(self.dim.tau_step_lengths, self.dim.days), dtype=np.float64)


class StandardSimulation(SimulationEventsMixin, MovementEventsMixin):
    """Runs singular simulation passes, producing time-series output."""

    _config: RumeConfig
    _params: ContextParams | None = None
    geo: Geo

    def __init__(self,
                 geo: Geo,
                 ipm: CompartmentModel,
                 mm: MovementModel | MovementSpec,
                 params: Params,
                 time_frame: TimeFrame,
                 initializer: Initializer | None = None,
                 rng: Callable[[], np.random.Generator] | None = None):
        SimulationEventsMixin.__init__(self)
        MovementEventsMixin.__init__(self)

        self.geo = geo
        if initializer is None:
            initializer = DEFAULT_INITIALIZER
        if rng is None:
            rng = np.random.default_rng

        self._config = RumeConfig(geo, ipm, mm, params, time_frame, initializer, rng)

    def validate(self) -> None:
        """Validate the simulation."""
        with error_gate("validating the simulation", ValidationException, CompilationException):
            ctx = RumeContext.from_config(self._config)
            check_attribute_declarations(ctx.ipm, ctx.mm)
            # ctx.validate_geo() # validate only the required geo parameters?
            validate_mm(ctx.mm.attributes, ctx.dim, ctx.geo, ctx.params)
            ctx.validate_ipm()
            # ctx.validate_init()

    @property
    def params(self) -> ContextParams:
        """Simulation parameters as used by this simulation."""
        # Here we lazily-evaluate and then cache params from the context.
        # Why not just cache the whole context when StandardSim is constructed? The problem is mutability.
        # Params is a dictionary, which allow mutation, and many of its values are
        # numpy arrays, which also allow mutation.
        # We can't really guarantee immutability in userland code or even our simulation code
        # so it's safest to reconstruct a fresh copy of the context every time we need it.
        # Of course, the user can still muck with this cached version of params, but the blast radius
        # for doing so is sufficiently contained by this approach because sim runs use a fresh context.
        # It would be nice to be able to deep-freeze the entire context object tree, but alas...
        if self._params is None:
            self._params = RumeContext.from_config(self._config).params
        return self._params

    def run(self) -> Output:
        """
        Run the simulation. It is safe to call this multiple times
        to run multiple independent simulations with the same configuraiton.
        """
        event_subs = Subscriber()

        with error_gate("compiling the simulation", CompilationException):
            ctx = RumeContext.from_config(self._config)
            ipm_exec = StandardIpmExecutor(ctx)
            movement_exec = StandardMovementExecutor(ctx)

            # Proxy the movement_exec's events, if anyone is listening for them.
            if MovementEventsMixin.has_subscribers(self):
                event_subs.subscribe(movement_exec.on_movement_start,
                                     self.on_movement_start.publish)
                event_subs.subscribe(movement_exec.on_movement_clause,
                                     self.on_movement_clause.publish)
                event_subs.subscribe(movement_exec.on_movement_finish,
                                     self.on_movement_finish.publish)

        with error_gate("initializing the simulation", InitException):
            ini = ctx.initialize()
            world = ListWorld.from_initials(ini)
            out = Output(ctx.dim, ctx.geo.labels.tolist(),
                         ctx.ipm.compartment_names, ctx.ipm.event_names, ini)

        self.on_start.publish(OnStart(dim=ctx.dim, time_frame=ctx.time_frame))

        for tick in ctx.clock():
            # First do movement
            with error_gate("executing the movement model", MmSimException, AttributeException):
                movement_exec.apply(world, tick)

            # Then do IPM
            with error_gate("executing the IPM", IpmSimException, AttributeException):
                tick_events, tick_prevalence = ipm_exec.apply(world, tick)
                out.incidence[tick.index] = tick_events
                out.prevalence[tick.index] = tick_prevalence

            t = tick.index
            self.on_tick.publish(OnTick(t, (t + 1) / ctx.dim.ticks))

        self.on_end.publish(None)

        event_subs.unsubscribe()
        return out


def check_attribute_declarations(ipm: CompartmentModel, mm: MovementModel | MovementSpec) -> None:
    """
    Check that the IPM's and MM's declared attributes are compatible.
    Raises ValidationException on any incompatibility.
    """
    # NOTE: for now, any overlapping declarations must be strictly equal,
    # but this should be relaxed when MMs support shape polymorphism.
    ipm_attrs = {a.name: a for a in ipm.attributes}
    mm_attrs = {a.name: a for a in mm.attributes}
    for name in set(ipm_attrs.keys()).intersection(mm_attrs.keys()):
        a1 = ipm_attrs[name]
        a2 = mm_attrs[name]
        if not a1.source == a2.source:
            msg = f"Both the IPM and MM declare attribute '{name}', but they are not from the same source ({a1.source} vs {a2.source})."
            raise ValidationException(msg)
        if not np.issubdtype(a1.dtype_as_np, a2.dtype):
            msg = f"Both the IPM and MM declare attribute '{name}', but they are not a type match ({a1.dtype_as_np} vs {np.dtype(a2.dtype)})."
            raise ValidationException(msg)
        if not a1.shape == a2.shape:
            msg = f"Both the IPM and MM declare attribute '{name}', but they are not a shape match ({a1.shape} vs {a2.shape})."
            raise ValidationException(msg)
