"""
Implements `BasicSimulator`: an epymorph simulation algorithm for performing single run,
forward simulations.
"""

from typing import Callable, Generic, TypeVar

import numpy as np

from epymorph.attribute import NamePattern
from epymorph.data_type import SimDType
from epymorph.error import (
    DataAttributeError,
    InitError,
    IPMSimError,
    MMSimError,
    SimCompilationError,
    SimValidationError,
    ValidationError,
    error_gate,
)
from epymorph.event import EventBus, OnStart, OnTick
from epymorph.rume import RUME
from epymorph.simulation import ParamValue, simulation_clock
from epymorph.simulator.basic.ipm_exec import IPMExecutor
from epymorph.simulator.basic.mm_exec import MovementExecutor
from epymorph.simulator.basic.output import Output
from epymorph.simulator.world_list import ListWorld
from epymorph.util import CovariantMapping

_events = EventBus()

RUMEType = TypeVar("RUMEType", bound=RUME)
"""A type of `RUME`."""


class BasicSimulator(Generic[RUMEType]):
    """
    A simulator for running singular simulation passes and producing time-series output.
    The most basic simulator.

    This class is generic in the type of RUME used to create it (`RUMEType`).

    Parameters
    ----------
    rume :
        The rume to use for the simulation.
    """

    rume: RUMEType
    """The RUME to use for the simulation."""
    ipm_exec: IPMExecutor
    """The class responsible for executing disease simulation."""
    mm_exec: MovementExecutor
    """The class responsible for executing movement simulation."""

    def __init__(self, rume: RUMEType):
        self.rume = rume

    def run(
        self,
        /,
        params: CovariantMapping[str | NamePattern, ParamValue] | None = None,
        rng_factory: Callable[[], np.random.Generator] | None = None,
    ) -> Output[RUMEType]:
        """
        Run a forward simulation on the RUME and produce one realization (output).

        Parameters
        ----------
        params :
            The set of parameters to override for the sake of this
            run. Specified as a dictionary of name/value pairs, where values
            can be in any form normally allowed for the construction of a RUME.
            Any parameter not overridden uses the value from the RUME.
        rng_factory :
            A function used to construct the random number generator for this
            run. This can be used to provide a seeded random number generator
            if consistent results are desired. By default `BasicSimulator` creates
            a new random number generator for each run, using numpy's default rng
            logic.

        Returns
        -------
        :
            The simulation results.
        """

        rume = self.rume

        rng = (rng_factory or np.random.default_rng)()

        with error_gate(
            "evaluating simulation attributes",
            ValidationError,
            SimCompilationError,
        ):
            try:
                data = rume.evaluate_params(override_params=params, rng=rng)
            except DataAttributeError as e:
                sub_es = e.exceptions if isinstance(e, ExceptionGroup) else (e,)
                err = "\n".join(
                    [
                        "RUME attribute requirements were not met. See errors:",
                        *(f"- {e}" for e in sub_es),
                    ]
                )
                raise SimValidationError(err) from None

            # Reject any parameter that evaluates to a masked numpy array.
            # This indicates unresolved data issues from an ADRIO.
            for key, value in data.raw_values.items():
                if np.ma.is_masked(value):
                    err = (
                        f"Parameter value {key} contains unresolved issues. Use ADRIO "
                        "constructor options to address all data issues as appropriate "
                        "before execution."
                    )
                    raise SimValidationError(err)

        with error_gate("initializing the simulation", InitError):
            initial_values = rume.initialize(data, rng)
            world = ListWorld.from_initials(initial_values)

        with error_gate("compiling the simulation", SimCompilationError):
            ipm_exec = IPMExecutor(rume, world, data, rng)
            movement_exec = MovementExecutor(rume, world, data, rng)

        name = self.__class__.__name__
        _events.on_start.publish(OnStart(name, rume))

        days = rume.time_frame.days
        taus = rume.num_tau_steps
        S = days * taus
        N = rume.scope.nodes
        C = rume.ipm.num_compartments
        E = rume.ipm.num_events
        visit_compartments = np.zeros((S, N, C), dtype=SimDType)
        visit_events = np.zeros((S, N, E), dtype=SimDType)
        home_compartments = np.zeros((S, N, C), dtype=SimDType)
        home_events = np.zeros((S, N, E), dtype=SimDType)

        # Run the simulation!
        for tick in simulation_clock(rume.time_frame, rume.tau_step_lengths):
            t = tick.sim_index

            # First do movement
            with error_gate("executing movement", MMSimError, DataAttributeError):
                movement_exec.apply(tick)

            # Then do IPM
            with error_gate("executing the IPM", IPMSimError, DataAttributeError):
                vcs, ves, hcs, hes = ipm_exec.apply(tick)
                visit_compartments[t] = vcs
                visit_events[t] = ves
                home_compartments[t] = hcs
                home_events[t] = hes

            _events.on_tick.publish(OnTick(t, S))

        _events.on_finish.publish(None)

        # Assemble output.
        return Output(
            rume=rume,
            initial=initial_values,
            visit_compartments=visit_compartments,
            visit_events=visit_events,
            home_compartments=home_compartments,
            home_events=home_events,
        )
