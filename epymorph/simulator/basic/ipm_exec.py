"""
`IPMExecutor` is part of the internal workings of `BasicSimulator`, implementing the
logic for processing the IPM phase of the simulation. Most users will not use this
module directly.
"""

from dataclasses import dataclass
from functools import reduce
from typing import ClassVar, Iterator, NamedTuple, Sequence

import numpy as np
from numpy.typing import NDArray

from epymorph.compartment_model import (
    BaseCompartmentModel,
    EdgeDef,
    ForkDef,
    TransitionDef,
    exogenous_states,
)
from epymorph.data_type import AttributeValue, SimArray, SimDType
from epymorph.database import DataResolver
from epymorph.error import (
    IPMSimInvalidForkError,
    IPMSimLessThanZeroError,
    IPMSimNaNError,
)
from epymorph.rume import RUME
from epymorph.simulation import Tick
from epymorph.simulator.world import Cohort, World
from epymorph.sympy_shim import SympyLambda, lambdify, lambdify_list
from epymorph.util import index_of


class Result(NamedTuple):
    """The result from executing a single IPM step."""

    visit_compartments: SimArray
    """updated compartments as a result of these events (an (N,C) array) tracked by the
    location where the individuals currently are"""
    visit_events: SimArray
    """events that happened this tick (an (N,E) array) tracked by the location where
    the event occurred"""
    home_compartments: SimArray
    """updated compartments as a result of these events (an (N,C) array) tracked by the
    location that the individuals consider home"""
    home_events: SimArray
    """events that happened this tick (an (N,E) array) tracked by the home location
    of the individuals effected"""


############################################################
# StandardIpmExecutor
############################################################


@dataclass(frozen=True)
class CompiledEdge:
    """Lambdified `EdgeDef` (no fork). Effectively: `poisson(rate * tau)`."""

    size: ClassVar[int] = 1
    """The number of edges in this transition."""
    rate_lambda: SympyLambda
    """The lambdified edge rate."""


@dataclass(frozen=True)
class CompiledFork:
    """Lambdified `ForkDef`. Effectively: `multinomial(poisson(rate * tau), prob)`."""

    size: int
    """The number of edges in this transition."""
    rate_lambda: SympyLambda
    """The lambdified edge base rate."""
    prob_lambda: SympyLambda
    """The lambdified edge fork probabilities."""


CompiledTransition = CompiledEdge | CompiledFork
"""A transition is either a fork or a plain edge."""


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


class IPMExecutor:
    """
    The standard implementation of compartment model IPM execution.

    Parameters
    ----------
    rume :
        The RUME.
    world :
        The world state.
    data :
        The evaluated data attributes of the simulation.
    rng :
        The simulation RNG.
    """

    _rume: RUME
    """The RUME."""
    _world: World
    """The world state."""
    _rng: np.random.Generator
    """The simulation RNG."""

    _trxs: list[CompiledTransition]
    """Compiled transitions."""
    _apply_matrix: NDArray[SimDType]
    """
    A matrix defining how each event impacts each compartment
    (subtracting or adding individuals).
    """
    _events_leaving_compartment: list[list[int]]
    """
    Mapping from compartment index to the list of event indices
    which source from that compartment.
    """
    _source_compartment_for_event: list[int]
    """Mapping from event index to the compartment index it sources from."""
    _attr_values: Iterator[list[AttributeValue]]
    """
    A generator for the list of arguments (from attributes) needed to evaluate
    transition functions.
    """

    def __init__(
        self,
        rume: RUME,
        world: World,
        data: DataResolver,
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
        self._rng = rng

        self._trxs = _compile_transitions(ipm)
        self._apply_matrix = _make_apply_matrix(ipm)
        self._events_leaving_compartment = events_leaving_compartment
        self._source_compartment_for_event = source_compartment_for_event
        self._attr_values = data.resolve_txn_series(
            ipm.requirements_dict.items(),
            rume.num_tau_steps,
        )

    def apply(self, tick: Tick) -> Result:
        """
        Apply the IPM for this tick, mutating the world state.

        Parameters
        ----------
        tick :
            Which tick to process.

        Returns
        -------
        :
            The location-specific events that happened this tick (an (N,E) array)
            and the new compartments resulting from these events (an (N,C) array).
        """
        N = self._rume.scope.nodes
        C = self._rume.ipm.num_compartments
        E = self._rume.ipm.num_events

        # (home_node, visit_node, :)
        events = np.zeros((N, N, E), dtype=SimDType)
        compartments = np.zeros((N, N, C), dtype=SimDType)

        for visit_node in range(N):
            # Sum all cohorts present in this node to get an effective total population.
            cohorts = self._world.get_cohorts(visit_node)
            effective = reduce(
                lambda a, b: a + b,
                map(lambda x: x.compartments, cohorts),
            )

            # Determine how many events happen in this tick.
            node_events = self._events(tick, visit_node, effective)  # (E,) array

            # Distribute events to the cohorts, proportional to their population.
            cohort_events = self._distribute(cohorts, node_events)  # (X,E) array

            # Now that events are assigned to cohorts,
            # convert to compartment deltas using apply matrix.
            self._world.apply_cohort_delta(
                visit_node,
                np.matmul(cohort_events, self._apply_matrix, dtype=SimDType),  # (X,C)
            )

            # Collect compartment/event info.
            for i, x in enumerate(cohorts):
                home = x.return_location
                events[home, visit_node, :] = cohort_events[i, :]
                compartments[home, visit_node, :] = x.compartments

        return Result(
            visit_compartments=compartments.sum(axis=0),
            visit_events=events.sum(axis=0),
            home_compartments=compartments.sum(axis=1),
            home_events=events.sum(axis=1),
        )

    def _events(self, tick: Tick, node: int, effective_pop: SimArray) -> SimArray:
        """
        Calculate how many events will happen this tick, correcting
        for the possibility of overruns. An (E,) array.
        """

        rate_args = [*effective_pop, *next(self._attr_values)]

        # Evaluate the event rates and do random draws for all transition events.
        occur = np.zeros(self._rume.ipm.num_events, dtype=SimDType)
        index = 0
        for t in self._trxs:
            try:
                match t:
                    case CompiledEdge(rate_lambda):
                        # get rate from lambda expression, catch divide by zero error
                        rate = rate_lambda(rate_args)
                        if rate < 0:
                            err = self._get_default_error_args(rate_args, node, tick)
                            raise IPMSimLessThanZeroError(err)
                        occur[index] = self._rng.poisson(rate * tick.tau)
                    case CompiledFork(size, rate_lambda, prob_lambda):
                        # get rate from lambda expression, catch divide by zero error
                        rate = rate_lambda(rate_args)
                        if rate < 0:
                            err = self._get_default_error_args(rate_args, node, tick)
                            raise IPMSimLessThanZeroError(err)
                        prob = prob_lambda(rate_args)
                        if any(n < 0 for n in prob):
                            err = self._get_invalid_prob_args(rate_args, node, tick, t)
                            raise IPMSimInvalidForkError(err)
                        occur[index : (index + size)] = self._rng.multinomial(
                            n=self._rng.poisson(rate * tick.tau),
                            pvals=prob,
                        )
                index += t.size
            except (ZeroDivisionError, FloatingPointError):
                err = self._get_zero_division_args(rate_args, node, tick, t)
                raise IPMSimNaNError(err) from None

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

    def _distribute(
        self,
        cohorts: Sequence[Cohort],
        events: NDArray[SimDType],  # (E,) array
    ) -> NDArray[SimDType]:
        """Distribute events across a location's cohorts. Returns an (X,E) result."""
        cohort_array = np.array(
            [x.compartments for x in cohorts],
            dtype=SimDType,
        )  # (X,C) array

        (X, _) = cohort_array.shape
        (E,) = events.shape

        # Each cohort is responsible for a proportion of the total population:
        total = cohort_array.sum()
        if total > 0:
            cohort_proportion = cohort_array.sum(axis=1) / total
        else:
            # If total population is zero, weight each cohort equally.
            cohort_proportion = np.ones(X) / X

        occurrences = np.zeros((X, E), dtype=SimDType)

        for eidx in range(E):
            occur: int = events[eidx]  # type: ignore
            cidx = self._source_compartment_for_event[eidx]
            if cidx == -1:
                # event is coming from an exogenous source
                # randomly distribute to cohorts based on their share of the population
                selected = self._rng.multinomial(
                    occur,
                    cohort_proportion,
                ).astype(SimDType)
            else:
                # event is coming from a modeled compartment
                selected = self._rng.multivariate_hypergeometric(
                    cohort_array[:, cidx],
                    occur,
                ).astype(SimDType)
                cohort_array[:, cidx] -= selected
            occurrences[:, eidx] = selected

        return occurrences

    def _get_default_error_args(
        self, rate_attrs: list, node: int, tick: Tick
    ) -> list[tuple[str, dict]]:
        """Assemble arguments error messages."""
        cvals = {
            name: value
            for name, value in zip(
                [c.name.full for c in self._rume.ipm.compartments],
                rate_attrs[: self._rume.ipm.num_compartments],
            )
        }
        pvals = {
            attribute.name: value
            for attribute, value in zip(
                self._rume.ipm.requirements,
                rate_attrs[self._rume.ipm.num_compartments :],
            )
        }
        return [
            ("Node : Timestep", {node: tick.step}),
            ("compartment values", cvals),
            ("ipm params", pvals),
        ]

    def _get_invalid_prob_args(
        self,
        rate_attrs: list,
        node: int,
        tick: Tick,
        transition: CompiledFork,
    ) -> list[tuple[str, dict]]:
        """Assemble arguments error messages."""
        arg_list = self._get_default_error_args(rate_attrs, node, tick)

        transition_index = self._trxs.index(transition)
        corr_transition = self._rume.ipm.transitions[transition_index]
        if isinstance(corr_transition, ForkDef):
            trx_vals = {
                str(corr_transition): corr_transition.rate,
                "Probabilities": ", ".join(
                    [str(expr) for expr in corr_transition.probs]
                ),
            }
            arg_list.append(
                ("corresponding fork transition and probabilities", trx_vals)
            )

        return arg_list

    def _get_zero_division_args(
        self,
        rate_attrs: list,
        node: int,
        tick: Tick,
        transition: CompiledEdge | CompiledFork,
    ) -> list[tuple[str, dict]]:
        """Assemble arguments for error messages."""
        arg_list = self._get_default_error_args(rate_attrs, node, tick)

        transition_index = self._trxs.index(transition)
        corr_transition = self._rume.ipm.transitions[transition_index]
        if isinstance(corr_transition, EdgeDef):
            trx = {str(corr_transition.name): corr_transition.rate}
            arg_list.append(("corresponding transition", trx))
        elif isinstance(corr_transition, ForkDef):
            trx = {str(corr_transition): corr_transition.rate}
            arg_list.append(("corresponding fork transition", trx))

        return arg_list
