"""Implements one epymorph simulation algorithm: the basic simulator."""
from typing import Callable, Mapping, cast

import numpy as np
from numpy.typing import NDArray

from epymorph.database import NamePattern
from epymorph.error import (AttributeException, CompilationException,
                            InitException, IpmSimException, MmSimException,
                            SimValidationException, ValidationException,
                            error_gate)
from epymorph.event import (MovementEventsMixin, OnStart, OnTick,
                            SimulationEventsMixin)
from epymorph.params import ParamValue
from epymorph.rume import GEO_LABELS, Rume
from epymorph.simulation import TimeFrame, simulation_clock
from epymorph.simulator.basic.ipm_exec import IpmExecutor
from epymorph.simulator.basic.mm_exec import MovementExecutor
from epymorph.simulator.basic.output import Output
from epymorph.simulator.data import (evaluate_params, initialize_rume,
                                     validate_attributes)
from epymorph.simulator.world_list import ListWorld


class BasicSimulator(SimulationEventsMixin, MovementEventsMixin):
    """
    A simulator for running singular simulation passes and producing time-series output.
    The most basic simulator!
    """

    rume: Rume
    ipm_exec: IpmExecutor
    mm_exec: MovementExecutor

    def __init__(self, rume: Rume):
        SimulationEventsMixin.__init__(self)
        MovementEventsMixin.__init__(self)
        self.rume = rume

    def run(
        self, /,
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

        with error_gate("evaluating simulation attributes", ValidationException, CompilationException):
            try:
                db = evaluate_params(
                    rume=rume,
                    override_params={
                        NamePattern.parse(k): v
                        for k, v in (params or {}).items()
                    },
                    rng=rng,
                )
                validate_attributes(rume, db)
            except AttributeException as e:
                msg = f"RUME attribute requirements were not met. See errors:\n- {e}"
                raise SimValidationException(msg) from None
            except ExceptionGroup as e:
                msg = "RUME attribute requirements were not met. See errors:" + \
                    "".join(f"\n- {e}" for e in e.exceptions)
                raise SimValidationException(msg) from None

        with error_gate("initializing the simulation", InitException):
            initial_values = initialize_rume(rume, rng, db)

            matched = db.query(GEO_LABELS)
            if matched is None:
                geo_labels = rume.scope.get_node_ids()
            else:
                geo_labels = cast(NDArray[np.str_], matched.value)

            out = Output(
                dim=dim,
                geo_labels=geo_labels.tolist(),
                compartment_labels=[c.name for c in rume.ipm.compartments],
                event_labels=rume.ipm.event_names,
                initial=initial_values,
            )

            world = ListWorld.from_initials(initial_values)

        with error_gate("compiling the simulation", CompilationException):
            ipm_exec = IpmExecutor(rume, world, db, rng)
            movement_exec = MovementExecutor(rume, world, db, rng, self)

        self.on_start.publish(OnStart(dim, rume.time_frame))

        # Run the simulation!
        for tick in simulation_clock(dim):
            # First do movement
            with error_gate("executing the movement model", MmSimException, AttributeException):
                movement_exec.apply(tick)

            # Then do IPM
            with error_gate("executing the IPM", IpmSimException, AttributeException):
                tick_events, tick_prevalence = ipm_exec.apply(tick)
                out.incidence[tick.sim_index] = tick_events
                out.prevalence[tick.sim_index] = tick_prevalence

            t = tick.sim_index
            self.on_tick.publish(OnTick(t, (t + 1) / dim.ticks))

        self.on_finish.publish(None)

        return out
