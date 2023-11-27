"""
IPM executor classes handle the logic for processing the IPM step of the simulation.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from epymorph.compartment_model import (EdgeDef, ForkDef, Transition,
                                        TransitionDef, exogenous_states)
from epymorph.engine.context import ExecutionContext, Tick
from epymorph.engine.world import World
from epymorph.simulation import SimDType
from epymorph.sympy_shim import Symbol, SympyLambda, lambdify, lambdify_list
from epymorph.util import index_where


class IpmExecutor(ABC):
    """
    Abstract interface responsible for advancing the simulation state due to the IPM.
    """

    @abstractmethod
    def apply(self, world: World, tick: Tick) -> tuple[NDArray[SimDType], NDArray[SimDType]]:
        """
        Applies the IPM for this tick, mutating the world state.
        Returns the tick's values of incidence and prevalence:
        - events that happened this tick (an (N,E) array), and
        - updated prevalence as a result of these events (an (N,C) array).
        """


############################################################
# StandardIpmExecutor
############################################################


@dataclass(frozen=True)
class _IndependentTrx:
    """Lambdified EdgeDef (no fork). Effectively: `poisson(rate * tau)`"""
    size: ClassVar[int] = 1
    rate_lambda: SympyLambda


@dataclass(frozen=True)
class _ForkedTrx:
    """Lambdified ForkDef. Effectively: `multinomial(poisson(rate * tau), prob)`"""
    size: int
    rate_lambda: SympyLambda
    prob_lambda: SympyLambda


_Trx = _IndependentTrx | _ForkedTrx


class StandardIpmExecutor(IpmExecutor):
    """The standard implementation of compartment model IPM execution."""

    _ctx: ExecutionContext
    """the sim context"""
    _trxs: list[_Trx]
    """compiled transitions"""
    _apply_matrix: NDArray[SimDType]
    """a matrix defining how each event impacts each compartment (subtracting or adding individuals)"""
    _events_leaving_compartment: list[list[int]]
    """mapping from compartment index to the list of event indices which source from that compartment"""
    _source_compartment_for_event: list[int]
    """mapping from event index to the compartment index it sources from"""

    def __init__(self, ctx: ExecutionContext):
        ipm = ctx.ipm
        transitions = ipm.transitions
        compartments = ipm.compartments
        attributes = ipm.attributes

        def compartment_index(s: Symbol) -> int:
            return index_where(compartments, lambda c: c.symbol == s)

        # Calc apply matrix -- values are {+1, 0, -1}
        E = ipm.num_events
        C = ipm.num_compartments
        apply_matrix = np.zeros((E, C), dtype=SimDType)
        for eidx, e in enumerate(Transition.as_events(transitions)):
            if e.compartment_from not in exogenous_states:
                apply_matrix[eidx, compartment_index(e.compartment_from)] = -1
            if e.compartment_to not in exogenous_states:
                apply_matrix[eidx, compartment_index(e.compartment_to)] = +1

        # Calc list of events leaving each compartment (each may have 0, 1, or more)
        events_leaving_compartment = [[eidx
                                       for eidx, e in enumerate(Transition.as_events(transitions))
                                       if e.compartment_from == c.symbol]
                                      for c in compartments]

        # Calc the source compartment for each event
        source_compartment_for_event = [compartment_index(e.compartment_from)
                                        for e in Transition.as_events(transitions)]

        # The parameters to pass to all rate lambdas
        rate_params = [*(c.symbol for c in compartments),
                       *(a.symbol for a in attributes)]

        def compile_transition(transition: TransitionDef) -> _Trx:
            match transition:
                case EdgeDef(rate, _, _):
                    rate_lambda = lambdify(rate_params, rate)
                    return _IndependentTrx(rate_lambda)
                case ForkDef(rate, edges, prob):
                    size = len(edges)
                    rate_lambda = lambdify(rate_params, rate)
                    prob_lambda = lambdify_list(rate_params, prob)
                    return _ForkedTrx(size, rate_lambda, prob_lambda)

        self._ctx = ctx
        self._trxs = [compile_transition(t) for t in transitions]
        self._apply_matrix = apply_matrix
        self._events_leaving_compartment = events_leaving_compartment
        self._source_compartment_for_event = source_compartment_for_event

    def apply(self, world: World, tick: Tick) -> tuple[NDArray[SimDType], NDArray[SimDType]]:
        """
        Applies the IPM for this tick, mutating the world state.
        Returns the location-specific events that happened this tick (an (N,E) array) and the new
        prevalence resulting from these events (an (N,C) array).
        """
        _, N, C, E = self._ctx.dim.TNCE
        tick_events = np.zeros((N, E), dtype=SimDType)
        tick_prevalence = np.zeros((N, C), dtype=SimDType)

        for node in range(N):
            cohorts = world.get_cohort_array(node)
            effective = cohorts.sum(axis=0, dtype=SimDType)

            occurrences = self._events(node, tick, effective)
            cohort_deltas = self._distribute(cohorts, occurrences)
            world.apply_cohort_delta(node, cohort_deltas)

            location_delta = cohort_deltas.sum(axis=0, dtype=SimDType)

            tick_events[node] = occurrences
            tick_prevalence[node] = effective + location_delta

        return tick_events, tick_prevalence

    def _events(self, node: int, tick: Tick, effective_pop: NDArray[SimDType]) -> NDArray[SimDType]:
        """Calculate how many events will happen this tick, correcting for the possibility of overruns."""
        rate_args = [*effective_pop,
                     *(self._ctx.get_attribute(a.attribute, tick, node)  # attribs
                       for a in self._ctx.ipm.attributes)]

        # Evaluate the event rates and do random draws for all transition events.
        occur = np.zeros(self._ctx.dim.events, dtype=SimDType)
        index = 0
        for t in self._trxs:
            match t:
                case _IndependentTrx(rate_lambda):
                    rate = rate_lambda(rate_args)
                    occur[index] = self._ctx.rng.poisson(rate * tick.tau)
                case _ForkedTrx(size, rate_lambda, prob_lambda):
                    rate = rate_lambda(rate_args)
                    base = self._ctx.rng.poisson(rate * tick.tau)
                    prob = prob_lambda(rate_args)
                    stop = index + size
                    occur[index:stop] = self._ctx.rng.multinomial(
                        base, prob)
            index += t.size

        # Check for event overruns leaving each compartment and correct counts.
        for cidx, eidxs in enumerate(self._events_leaving_compartment):
            available = effective_pop[cidx]
            n = len(eidxs)
            if n == 0:
                # Compartment has no outward edges; nothing to do here.
                continue
            elif n == 1:
                # Compartment only has one outward edge; just "min".
                eidx = eidxs[0]
                occur[eidx] = min(occur[eidx], available)
            elif n == 2:
                # Compartment has two outward edges:
                # use hypergeo to select which events "actually" happened.
                desired0, desired1 = occur[eidxs]
                if desired0 + desired1 > available:
                    drawn0 = self._ctx.rng.hypergeometric(
                        desired0, desired1, available)
                    occur[eidxs] = [drawn0, available - drawn0]
            else:
                # Compartment has more than two outwards edges:
                # use multivariate hypergeometric to select which events "actually" happened.
                desired = occur[eidxs]
                if np.sum(desired) > available:
                    occur[eidxs] = self._ctx.rng.multivariate_hypergeometric(
                        desired, available)
        return occur

    def _distribute(self, cohorts: NDArray[SimDType], events: NDArray[SimDType]) -> NDArray[SimDType]:
        """Distribute all events across a location's cohorts and return the compartment deltas for each."""
        x = cohorts.shape[0]
        e = self._ctx.dim.events
        occurrences = np.zeros((x, e), dtype=SimDType)
        for eidx in range(e):
            occur: int = events[eidx]  # type: ignore
            cidx = self._source_compartment_for_event[eidx]
            if cidx == -1:
                # event is coming from an exogenous source
                occurrences[:, eidx] = occur
            else:
                # event is coming from a modeled compartment
                selected = self._ctx.rng.multivariate_hypergeometric(
                    cohorts[:, cidx],
                    occur
                ).astype(SimDType)
                occurrences[:, eidx] = selected
                cohorts[:, cidx] -= selected

        # Now that events are assigned to pops, convert to compartment deltas using apply matrix.
        return np.matmul(occurrences, self._apply_matrix, dtype=SimDType)
