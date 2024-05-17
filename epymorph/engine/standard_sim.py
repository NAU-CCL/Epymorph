"""
The most fundamental epymorph simulation type:
run a single simulation from start to finish with a static set of parameters.
"""
from typing import Callable

import numpy as np
from typing_extensions import deprecated

from epymorph.compartment_model import CompartmentModel, validate_ipm
from epymorph.data_shape import SimDimensions
from epymorph.engine.context import RumeContext
from epymorph.engine.ipm_exec import StandardIpmExecutor
from epymorph.engine.mm_exec import StandardMovementExecutor
from epymorph.engine.output import Output
from epymorph.engine.world_list import ListWorld
from epymorph.error import (AttributeException, CompilationException,
                            InitException, IpmSimException, MmSimException,
                            ValidationException, error_gate)
from epymorph.event import (MovementEventsMixin, OnStart, OnTick,
                            SimulationEventsMixin)
from epymorph.geo.geo import Geo
from epymorph.initializer import DEFAULT_INITIALIZER, Initializer, initialize
from epymorph.movement.movement_model import MovementModel, validate_mm
from epymorph.movement.parser import MovementSpec
from epymorph.params import NormalizedParamsDict, RawParams, normalize_params
from epymorph.simulation import TimeFrame
from epymorph.util import Subscriber


class StandardSimulation(SimulationEventsMixin, MovementEventsMixin):
    """Runs singular simulation passes, producing time-series output."""

    dim: SimDimensions
    geo: Geo
    ipm: CompartmentModel
    mm: MovementSpec
    initializer: Initializer
    time_frame: TimeFrame
    rng_factory: Callable[[], np.random.Generator]
    _raw_params: RawParams

    def __init__(self,
                 geo: Geo,
                 ipm: CompartmentModel,
                 mm: MovementSpec,
                 params: RawParams,
                 time_frame: TimeFrame,
                 initializer: Initializer | None = None,
                 rng: Callable[[], np.random.Generator] | None = None):
        SimulationEventsMixin.__init__(self)
        MovementEventsMixin.__init__(self)

        self.dim = SimDimensions.build(
            tau_step_lengths=mm.steps.step_lengths,
            start_date=time_frame.start_date,
            days=time_frame.duration_days,
            nodes=geo.nodes,
            compartments=ipm.num_compartments,
            events=ipm.num_events,
        )
        self.geo = geo
        self.ipm = ipm
        self.mm = mm
        self.initializer = initializer or DEFAULT_INITIALIZER
        self.time_frame = time_frame
        self.rng_factory = rng or np.random.default_rng
        self._raw_params = params

    @deprecated("Validation will happen automatically in future during run.")
    def validate(self) -> None:
        """Validate the simulation."""
        with error_gate("validating the simulation", ValidationException, CompilationException):
            # (See comments in validation step during `run()`)
            norm_params = normalize_params(
                self._raw_params, self.geo, self.dim, dtypes=self.ipm.attribute_dtypes)
            check_attribute_declarations(self.ipm, self.mm)
            validate_mm(self.mm.attributes, self.dim, self.geo, norm_params)
            validate_ipm(self.ipm, self.dim, self.geo, norm_params)

    @property
    def params(self) -> NormalizedParamsDict:
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
        return normalize_params(self._raw_params, self.geo, self.dim, dtypes=self.ipm.attribute_dtypes)

    def run(self) -> Output:
        """
        Run the simulation. It is safe to call this multiple times
        to run multiple independent simulations with the same configuraiton.
        """
        with error_gate("preparing the simulation context", ValidationException, CompilationException):
            norm_params = normalize_params(
                self._raw_params, self.geo, self.dim, dtypes=self.ipm.attribute_dtypes)

            ctx = RumeContext(
                dim=self.dim,
                geo=self.geo,
                ipm=self.ipm,
                rng=self.rng_factory(),
                params=norm_params,
            )

        with error_gate("validating the simulation", ValidationException, CompilationException):
            # Each strata IPM and corresponding MM should be attribute-compatible.
            # We can't check against the combined IPM, because the attribute names
            # are remapped in the combined IPM. You might think this makes attribute
            # compatibility unnecessary, but it still is, because of the way params
            # are provided to the simulation.
            check_attribute_declarations(self.ipm, self.mm)
            # Then we can validate the MM's params.
            validate_mm(self.mm.attributes, self.dim, self.geo, norm_params)
            validate_ipm(self.ipm, self.dim, self.geo, norm_params)
            # TODO: more validation!
            # ctx.validate_geo() # validate only the required geo parameters?
            # ctx.validate_init()

        event_subs = Subscriber()

        with error_gate("compiling the simulation", CompilationException):
            ipm_exec = StandardIpmExecutor(ctx, self.ipm)
            movement_exec = StandardMovementExecutor(ctx, self.mm)

            # Proxy the movement_exec's events, if anyone is listening for them.
            if MovementEventsMixin.has_subscribers(self):
                event_subs.subscribe(movement_exec.on_movement_start,
                                     self.on_movement_start.publish)
                event_subs.subscribe(movement_exec.on_movement_clause,
                                     self.on_movement_clause.publish)
                event_subs.subscribe(movement_exec.on_movement_finish,
                                     self.on_movement_finish.publish)

        with error_gate("initializing the simulation", InitException):
            init = initialize(
                self.initializer,
                self.dim,
                self.geo,
                self._raw_params,
                ctx.rng,
            )
            world = ListWorld.from_initials(init)
            out = Output(
                self.dim,
                self.geo.labels.tolist(),
                self.ipm.compartment_names,
                self.ipm.event_names,
                init,
            )

        self.on_start.publish(OnStart(dim=ctx.dim, time_frame=self.time_frame))

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
