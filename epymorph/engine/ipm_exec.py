"""
IPM executor classes handle the logic for processing the IPM step of the simulation.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Protocol

import numpy as np
from numpy.typing import NDArray

from epymorph.compartment_model import (CompartmentModel, EdgeDef, ForkDef,
                                        TransitionDef, exogenous_states)
from epymorph.data_shape import SimDimensions
from epymorph.data_type import AttributeScalar, SimArray, SimDType
from epymorph.engine.world import World
from epymorph.error import (IpmSimInvalidProbsException,
                            IpmSimLessThanZeroException, IpmSimNaNException)
from epymorph.simulation import AttributeDef, Tick
from epymorph.sympy_shim import SympyLambda, lambdify, lambdify_list
from epymorph.util import index_of


class IpmExecutor(ABC):
    """
    Abstract interface responsible for advancing the simulation state due to the IPM.
    """

    @abstractmethod
    def apply(self, world: World, tick: Tick) -> tuple[SimArray, SimArray]:
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


def _make_apply_matrix(ipm: CompartmentModel) -> SimArray:
    """
    Calc apply matrix; this matrix is used to apply a set of events
    to the compartments they impact. In general, an event indicates
    a transition from one state to another, so it is subtracted from one
    and added to the other. Events involving exogenous states, however,
    either add or subtract from the model but not both. By nature, they
    alter the number of individuals in the model. Matrix values are {+1, 0, -1}.
    """
    csymbols = [c.symbol for c in ipm.compartments]
    matrix_size = (ipm.num_events, ipm.num_compartments)
    apply_matrix = np.zeros(matrix_size, dtype=SimDType)
    for eidx, e in enumerate(ipm.events):
        if e.compartment_from not in exogenous_states:
            apply_matrix[eidx, index_of(csymbols, e.compartment_from)] = -1
        if e.compartment_to not in exogenous_states:
            apply_matrix[eidx, index_of(csymbols, e.compartment_to)] = +1
    return apply_matrix


class IpmContext(Protocol):
    """The subset of RumeContext that the IPM executor needs."""
    @property
    def dim(self) -> SimDimensions:
        """The simulation's dimensionality."""
        raise NotImplementedError

    @property
    def rng(self) -> np.random.Generator:
        """The random number generator."""
        raise NotImplementedError

    @abstractmethod
    def get_attribute(self, attr: AttributeDef, tick: Tick, node: int) -> AttributeScalar:
        """Get an attribute value at a specific tick and node."""
        raise NotImplementedError


class StandardIpmExecutor(IpmExecutor):
    """The standard implementation of compartment model IPM execution."""

    _ctx: IpmContext
    """the sim context"""
    _ipm: CompartmentModel
    """the IPM"""
    _trxs: list[_Trx]
    """compiled transitions"""
    _apply_matrix: NDArray[SimDType]
    """a matrix defining how each event impacts each compartment (subtracting or adding individuals)"""
    _events_leaving_compartment: list[list[int]]
    """mapping from compartment index to the list of event indices which source from that compartment"""
    _source_compartment_for_event: list[int]
    """mapping from event index to the compartment index it sources from"""

    def __init__(self, ctx: IpmContext, ipm: CompartmentModel):
        # Calc list of events leaving each compartment (each may have 0, 1, or more)
        events_leaving_compartment = [[eidx
                                       for eidx, e in enumerate(ipm.events)
                                       if e.compartment_from == c.symbol]
                                      for c in ipm.compartments]

        # Calc the source compartment for each event
        csymbols = [c.symbol for c in ipm.compartments]
        source_compartment_for_event = [index_of(csymbols, e.compartment_from)
                                        for e in ipm.events]

        # The parameters to pass to all rate lambdas
        rate_params = [*csymbols, *(a.symbol for a in ipm.attributes)]

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
        self._ipm = ipm
        self._trxs = [compile_transition(t) for t in ipm.transitions]
        self._apply_matrix = _make_apply_matrix(ipm)
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
                     *(self._ctx.get_attribute(a, tick, node)  # attribs
                       for a in self._ipm.attributes)]

        # Evaluate the event rates and do random draws for all transition events.
        occur = np.zeros(self._ctx.dim.events, dtype=SimDType)
        index = 0
        for t in self._trxs:
            match t:
                case _IndependentTrx(rate_lambda):
                    # get rate from lambda expression, catch divide by zero error
                    try:
                        rate = rate_lambda(rate_args)
                    except (ZeroDivisionError, FloatingPointError):
                        raise IpmSimNaNException(
                            self._get_zero_division_args(
                                rate_args, node, tick, t)
                        ) from None
                    # check for < 0 rate, throw error in this case
                    if rate < 0:
                        raise IpmSimLessThanZeroException(
                            self._get_default_error_args(rate_args, node, tick)
                        )
                    occur[index] = self._ctx.rng.poisson(rate * tick.tau)
                case _ForkedTrx(size, rate_lambda, prob_lambda):
                    # get rate from lambda expression, catch divide by zero error
                    try:
                        rate = rate_lambda(rate_args)
                    except (ZeroDivisionError, FloatingPointError):
                        raise IpmSimNaNException(
                            self._get_zero_division_args(
                                rate_args, node, tick, t)
                        ) from None
                    # check for < 0 base, throw error in this case
                    if rate < 0:
                        raise IpmSimLessThanZeroException(
                            self._get_default_error_args(rate_args, node, tick)
                        )
                    base = self._ctx.rng.poisson(rate * tick.tau)
                    prob = prob_lambda(rate_args)
                    # check for negative probs
                    if any(n < 0 for n in prob):
                        raise IpmSimInvalidProbsException(
                            self._get_invalid_prob_args(rate_args, node, tick, t)
                        )
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

    def _get_default_error_args(self, rate_attrs: list, node: int, tick: Tick) -> list[tuple[str, dict]]:
        arg_list = []
        arg_list.append(("Node : Timestep", {node: tick.step}))
        arg_list.append(("compartment values", {
            name: value for (name, value) in zip(self._ipm.compartment_names,
                                                 rate_attrs[:self._ctx.dim.compartments])
        }))
        arg_list.append(("ipm params", {
            attribute.name: value for (attribute, value) in zip(self._ipm.attributes,
                                                                rate_attrs[self._ctx.dim.compartments:])
        }))

        return arg_list

    def _get_invalid_prob_args(self, rate_attrs: list, node: int, tick: Tick,
                               transition: _ForkedTrx) -> list[tuple[str, dict]]:
        arg_list = self._get_default_error_args(rate_attrs, node, tick)

        transition_index = self._trxs.index(transition)
        corr_transition = self._ipm.transitions[transition_index]
        if isinstance(corr_transition, ForkDef):
            to_compartments = ", ".join([str(edge.compartment_to)
                                        for edge in corr_transition.edges])
            from_compartment = corr_transition.edges[0].compartment_from
            arg_list.append(("corresponding fork transition and probabilities",
                             {
                                 f"{from_compartment}->({to_compartments})": corr_transition.rate,
                                 "Probabilities": ', '.join([str(expr) for expr in corr_transition.probs]),
                             }))

        return arg_list

    def _get_zero_division_args(self, rate_attrs: list, node: int, tick: Tick,
                                transition: _IndependentTrx | _ForkedTrx) -> list[tuple[str, dict]]:
        arg_list = self._get_default_error_args(rate_attrs, node, tick)

        transition_index = self._trxs.index(transition)
        corr_transition = self._ipm.transitions[transition_index]
        if isinstance(corr_transition, EdgeDef):
            arg_list.append(("corresponding transition", {
                            f"{corr_transition.compartment_from}->{corr_transition.compartment_to}": corr_transition.rate}))
        if isinstance(corr_transition, ForkDef):
            to_compartments = ", ".join([str(edge.compartment_to)
                                        for edge in corr_transition.edges])
            from_compartment = corr_transition.edges[0].compartment_from
            arg_list.append(("corresponding fork transition", {
                            f"{from_compartment}->({to_compartments})": corr_transition.rate}))

        return arg_list

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
