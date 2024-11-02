"""
IPM executor classes handle the logic for processing the IPM step of the simulation.
"""

from dataclasses import dataclass
from typing import ClassVar, Generator, Iterable, NamedTuple

import numpy as np
from numpy.typing import NDArray

from epymorph.compartment_model import (
    BaseCompartmentModel,
    EdgeDef,
    ForkDef,
    TransitionDef,
    exogenous_states,
)
from epymorph.data_type import AttributeArray, AttributeValue, SimArray, SimDType
from epymorph.database import Database
from epymorph.error import (
    IpmSimInvalidProbsException,
    IpmSimLessThanZeroException,
    IpmSimNaNException,
)
from epymorph.rume import Rume
from epymorph.simulation import AttributeResolver, Tick
from epymorph.simulator.world import World
from epymorph.sympy_shim import SympyLambda, lambdify, lambdify_list
from epymorph.util import index_of


class Result(NamedTuple):
    """The result from executing a single IPM step."""

    events: SimArray
    """events that happened this tick (an (N,E) array)"""
    compartments: SimArray
    """updated compartments as a result of these events (an (N,C) array)."""


############################################################
# StandardIpmExecutor
############################################################


@dataclass(frozen=True)
class CompiledEdge:
    """Lambdified EdgeDef (no fork). Effectively: `poisson(rate * tau)`"""

    size: ClassVar[int] = 1
    rate_lambda: SympyLambda


@dataclass(frozen=True)
class CompiledFork:
    """Lambdified ForkDef. Effectively: `multinomial(poisson(rate * tau), prob)`"""

    size: int
    rate_lambda: SympyLambda
    prob_lambda: SympyLambda


CompiledTransition = CompiledEdge | CompiledFork


def _compile_transitions(model: BaseCompartmentModel) -> list[CompiledTransition]:
    # The parameters to pass to all rate lambdas
    rate_params = [*model.symbols.all_compartments, *model.symbols.all_requirements]

    def f(transition: TransitionDef) -> CompiledTransition:
        match transition:
            case EdgeDef(_, rate, _, _):
                rate_lambda = lambdify(rate_params, rate)
                return CompiledEdge(rate_lambda)
            case ForkDef(rate, edges, prob):
                size = len(edges)
                rate_lambda = lambdify(rate_params, rate)
                prob_lambda = lambdify_list(rate_params, prob)
                return CompiledFork(size, rate_lambda, prob_lambda)

    return [f(t) for t in model.transitions]


def _make_apply_matrix(ipm: BaseCompartmentModel) -> SimArray:
    """
    Calc apply matrix; this matrix is used to apply a set of events
    to the compartments they impact. In general, an event indicates
    a transition from one state to another, so it is subtracted from one
    and added to the other. Events involving exogenous states, however,
    either add or subtract from the model but not both. By nature, they
    alter the number of individuals in the model. Matrix values are {+1, 0, -1}.
    """
    csymbols = ipm.symbols.all_compartments
    matrix_size = (ipm.num_events, ipm.num_compartments)
    apply_matrix = np.zeros(matrix_size, dtype=SimDType)
    for eidx, e in enumerate(ipm.events):
        if e.compartment_from not in exogenous_states:
            apply_matrix[eidx, index_of(csymbols, e.compartment_from)] = -1
        if e.compartment_to not in exogenous_states:
            apply_matrix[eidx, index_of(csymbols, e.compartment_to)] = +1
    return apply_matrix


class IpmExecutor:
    """The standard implementation of compartment model IPM execution."""

    _rume: Rume
    """the RUME"""
    _world: World
    """the world state"""
    _data: Database[AttributeArray]
    """resolver for simulation data"""
    _rng: np.random.Generator
    """the simulation RNG"""

    _trxs: list[CompiledTransition]
    """compiled transitions"""
    _apply_matrix: NDArray[SimDType]
    """
    a matrix defining how each event impacts each compartment
    (subtracting or adding individuals)
    """
    _events_leaving_compartment: list[list[int]]
    """
    mapping from compartment index to the list of event indices
    which source from that compartment
    """
    _source_compartment_for_event: list[int]
    """mapping from event index to the compartment index it sources from"""
    _attribute_values_txn: Generator[Iterable[AttributeValue], None, None]
    """
    a generator for the list of arguments (from attributes) needed to evaluate
    transition functions
    """

    def __init__(
        self,
        rume: Rume,
        world: World,
        data: Database[AttributeArray],
        rng: np.random.Generator,
    ):
        ipm = rume.ipm
        csymbols = ipm.symbols.all_compartments

        # Calc list of events leaving each compartment (each may have 0, 1, or more)
        events_leaving_compartment = [
            [eidx for eidx, e in enumerate(ipm.events) if e.compartment_from == c]
            for c in csymbols
        ]

        # Calc the source compartment for each event
        source_compartment_for_event = [
            index_of(csymbols, e.compartment_from) for e in ipm.events
        ]

        self._rume = rume
        self._world = world
        self._data = data
        self._rng = rng

        self._trxs = _compile_transitions(ipm)
        self._apply_matrix = _make_apply_matrix(ipm)
        self._events_leaving_compartment = events_leaving_compartment
        self._source_compartment_for_event = source_compartment_for_event
        self._attribute_values_txn = AttributeResolver(
            data, rume.dim
        ).resolve_txn_series(list(ipm.requirements_dict.items()))

    def apply(self, tick: Tick) -> Result:
        """
        Applies the IPM for this tick, mutating the world state.
        Returns the location-specific events that happened this tick (an (N,E) array)
        and the new compartments resulting from these events (an (N,C) array).
        """
        _, N, C, E = self._rume.dim.TNCE
        tick_events = np.zeros((N, E), dtype=SimDType)
        tick_compartments = np.zeros((N, C), dtype=SimDType)

        for node in range(N):
            cohorts = self._world.get_cohort_array(node)
            effective = cohorts.sum(axis=0, dtype=SimDType)

            occurrences = self._events(tick, node, effective)
            cohort_deltas = self._distribute(cohorts, occurrences)
            self._world.apply_cohort_delta(node, cohort_deltas)

            location_delta = cohort_deltas.sum(axis=0, dtype=SimDType)

            tick_events[node] = occurrences
            tick_compartments[node] = effective + location_delta

        return Result(tick_events, tick_compartments)

    def _events(self, tick: Tick, node: int, effective_pop: SimArray) -> SimArray:
        """
        Calculate how many events will happen this tick, correcting
        for the possibility of overruns.
        """

        rate_args = [*effective_pop, *next(self._attribute_values_txn)]

        # Evaluate the event rates and do random draws for all transition events.
        occur = np.zeros(self._rume.dim.events, dtype=SimDType)
        index = 0
        for t in self._trxs:
            match t:
                case CompiledEdge(rate_lambda):
                    # get rate from lambda expression, catch divide by zero error
                    try:
                        rate = rate_lambda(rate_args)
                    except (ZeroDivisionError, FloatingPointError):
                        raise IpmSimNaNException(
                            self._get_zero_division_args(rate_args, node, tick, t)
                        ) from None
                    # check for < 0 rate, throw error in this case
                    if rate < 0:
                        raise IpmSimLessThanZeroException(
                            self._get_default_error_args(rate_args, node, tick)
                        )
                    occur[index] = self._rng.poisson(rate * tick.tau)
                case CompiledFork(size, rate_lambda, prob_lambda):
                    # get rate from lambda expression, catch divide by zero error
                    try:
                        rate = rate_lambda(rate_args)
                    except (ZeroDivisionError, FloatingPointError):
                        raise IpmSimNaNException(
                            self._get_zero_division_args(rate_args, node, tick, t)
                        ) from None
                    # check for < 0 base, throw error in this case
                    if rate < 0:
                        raise IpmSimLessThanZeroException(
                            self._get_default_error_args(rate_args, node, tick)
                        )
                    base = self._rng.poisson(rate * tick.tau)
                    prob = prob_lambda(rate_args)
                    # check for negative probs
                    if any(n < 0 for n in prob):
                        raise IpmSimInvalidProbsException(
                            self._get_invalid_prob_args(rate_args, node, tick, t)
                        )
                    stop = index + size
                    occur[index:stop] = self._rng.multinomial(base, prob)
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
                    drawn0 = self._rng.hypergeometric(desired0, desired1, available)
                    occur[eidxs] = [drawn0, available - drawn0]
            else:
                # Compartment has more than two outwards edges:
                # use multivariate hypergeometric to select which events
                # "actually" happened.
                desired = occur[eidxs]
                if np.sum(desired) > available:
                    occur[eidxs] = self._rng.multivariate_hypergeometric(
                        desired, available
                    )
        return occur

    def _get_default_error_args(
        self, rate_attrs: list, node: int, tick: Tick
    ) -> list[tuple[str, dict]]:
        arg_list = []
        arg_list.append(("Node : Timestep", {node: tick.step}))
        arg_list.append(
            (
                "compartment values",
                {
                    name: value
                    for name, value in zip(
                        [c.name.full for c in self._rume.ipm.compartments],
                        rate_attrs[: self._rume.dim.compartments],
                    )
                },
            )
        )
        arg_list.append(
            (
                "ipm params",
                {
                    attribute.name: value
                    for attribute, value in zip(
                        self._rume.ipm.requirements,
                        rate_attrs[self._rume.dim.compartments :],
                    )
                },
            )
        )

        return arg_list

    def _get_invalid_prob_args(
        self, rate_attrs: list, node: int, tick: Tick, transition: CompiledFork
    ) -> list[tuple[str, dict]]:
        arg_list = self._get_default_error_args(rate_attrs, node, tick)

        transition_index = self._trxs.index(transition)
        corr_transition = self._rume.ipm.transitions[transition_index]
        if isinstance(corr_transition, ForkDef):
            to_compartments = ", ".join(
                [str(edge.compartment_to) for edge in corr_transition.edges]
            )
            from_compartment = corr_transition.edges[0].compartment_from
            arg_list.append(
                (
                    "corresponding fork transition and probabilities",
                    {
                        f"{from_compartment}->({to_compartments})": corr_transition.rate,  # noqa: E501
                        "Probabilities": ", ".join(
                            [str(expr) for expr in corr_transition.probs]
                        ),
                    },
                )
            )

        return arg_list

    def _get_zero_division_args(
        self,
        rate_attrs: list,
        node: int,
        tick: Tick,
        transition: CompiledEdge | CompiledFork,
    ) -> list[tuple[str, dict]]:
        arg_list = self._get_default_error_args(rate_attrs, node, tick)

        transition_index = self._trxs.index(transition)
        corr_transition = self._rume.ipm.transitions[transition_index]
        if isinstance(corr_transition, EdgeDef):
            arg_list.append(
                (
                    "corresponding transition",
                    {
                        f"{corr_transition.compartment_from}->{corr_transition.compartment_to}": corr_transition.rate  # noqa: E501
                    },
                )
            )
        if isinstance(corr_transition, ForkDef):
            to_compartments = ", ".join(
                [str(edge.compartment_to) for edge in corr_transition.edges]
            )
            from_compartment = corr_transition.edges[0].compartment_from
            arg_list.append(
                (
                    "corresponding fork transition",
                    {f"{from_compartment}->({to_compartments})": corr_transition.rate},
                )
            )

        return arg_list

    def _distribute(
        self, cohorts: NDArray[SimDType], events: NDArray[SimDType]
    ) -> NDArray[SimDType]:
        """
        Distribute all events across a location's cohorts and return
        the compartment deltas for each.
        """
        x = cohorts.shape[0]
        e = self._rume.dim.events
        occurrences = np.zeros((x, e), dtype=SimDType)
        for eidx in range(e):
            occur: int = events[eidx]  # type: ignore
            cidx = self._source_compartment_for_event[eidx]
            if cidx == -1:
                # event is coming from an exogenous source
                occurrences[:, eidx] = occur
            else:
                # event is coming from a modeled compartment
                selected = self._rng.multivariate_hypergeometric(
                    cohorts[:, cidx], occur
                ).astype(SimDType)
                occurrences[:, eidx] = selected
                cohorts[:, cidx] -= selected

        # Now that events are assigned to pops,
        # convert to compartment deltas using apply matrix.
        return np.matmul(occurrences, self._apply_matrix, dtype=SimDType)
