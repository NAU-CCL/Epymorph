# ruff: noqa: ERA001 # TODO: remove this
from dataclasses import dataclass
from functools import reduce
from typing import Callable, ClassVar, Generic, Sequence, TypeVar

import numpy as np
from numpy.typing import NDArray

from epymorph.attribute import ModuleNamespace
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
    DataAttributeError,
    InitError,
    IPMSimError,
    IPMSimInvalidForkError,
    IPMSimLessThanZeroError,
    IPMSimNaNError,
    MMSimError,
    SimCompilationError,
    SimValidationError,
    ValidationError,
    error_gate,
)
from epymorph.event import EventBus, OnStart, OnTick
from epymorph.movement_model import MovementClause
from epymorph.rume import RUME
from epymorph.simulation import Context, Tick, resolve_tick_delta, simulation_clock
from epymorph.simulator.basic.mm_exec import calculate_travelers
from epymorph.simulator.basic.output import Output
from epymorph.simulator.world import Cohort, World
from epymorph.simulator.world_list import ListWorld
from epymorph.strata import gpm_strata
from epymorph.sympy_shim import SympyLambda, lambdify, lambdify_list
from epymorph.util import index_of

RUMEType = TypeVar("RUMEType", bound=RUME)
"""A type of `RUME`."""


def rume_dims(rume: RUME) -> tuple[int, int, int, int, int]:
    T = rume.time_frame.days
    S = T * rume.num_tau_steps
    N = rume.scope.nodes
    C = rume.ipm.num_compartments
    E = rume.ipm.num_events
    return T, S, N, C, E


class FunctionSimulator(Generic[RUMEType]):
    rume: RUMEType
    """The RUME to use for the simulation."""

    def __init__(self, rume: RUMEType):
        self.rume = rume

    @staticmethod
    def evaluate_params(
        rume: RUME,
        rng: np.random.Generator,
    ) -> DataResolver:
        try:
            data = rume.evaluate_params(rng=rng)
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

        return data

    def run(self, rng: np.random.Generator | None = None) -> Output[RUMEType]:
        [params_rng, init_rng, run_rng] = (rng or np.random.default_rng()).spawn(3)

        with error_gate("evaluating attributes", ValidationError, SimCompilationError):
            data = FunctionSimulator.evaluate_params(self.rume, params_rng)

        with error_gate("compiling the simulation", SimCompilationError):
            ipm_exec = IPMExecutor.from_rume(self.rume, data)
            movement_exec = MovementExecutor.from_rume(self.rume, data)

        with error_gate("initializing the simulation", InitError):
            initial = self.rume.initialize(data, init_rng)
            world = ListWorld.from_initials(initial)

        T, S, N, C, E = rume_dims(self.rume)
        visit_compartments = np.zeros((S, N, C), dtype=SimDType)
        visit_events = np.zeros((S, N, E), dtype=SimDType)
        home_compartments = np.zeros((S, N, C), dtype=SimDType)
        home_events = np.zeros((S, N, E), dtype=SimDType)

        def save_tick_results(tick: int, cs: SimArray, es: SimArray) -> None:
            visit_compartments[tick] = cs.sum(axis=0)
            visit_events[tick] = es.sum(axis=0)
            home_compartments[tick] = cs.sum(axis=1)
            home_events[tick] = es.sum(axis=1)

        self.run_one(
            world=world,
            ipm_exec=ipm_exec,
            movement_exec=movement_exec,
            save_tick_results=save_tick_results,
            rng=run_rng,
        )

        return Output(
            rume=self.rume,
            initial=initial,
            visit_compartments=visit_compartments,
            visit_events=visit_events,
            home_compartments=home_compartments,
            home_events=home_events,
        )

    def run_one(
        self,
        world: World,
        ipm_exec: "IPMExecutor",
        movement_exec: "MovementExecutor",
        save_tick_results: Callable[[int, SimArray, SimArray], None],
        rng: np.random.Generator,
    ) -> None:
        event_bus = EventBus()
        S = self.rume.num_ticks
        name = self.__class__.__name__
        event_bus.on_start.publish(OnStart(name, self.rume))

        clock = list(simulation_clock(self.rume.time_frame, self.rume.tau_step_lengths))
        step_movement = movement_exec.stepper(clock, world, rng)
        step_ipm = ipm_exec.stepper(clock, world, save_tick_results, rng)
        for tick in clock:
            with error_gate("executing movement", MMSimError, DataAttributeError):
                step_movement()
            with error_gate("executing the IPM", IPMSimError, DataAttributeError):
                step_ipm()
            event_bus.on_tick.publish(OnTick(tick.sim_index, S))
        event_bus.on_finish.publish(None)


###############
# IPMExecutor #
###############


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


@dataclass(frozen=True)
class IPMExecutor:
    """
    The standard implementation of compartment model IPM execution.

    Parameters
    ----------
    rume :
        The RUME.
    data :
        The evaluated data attributes of the simulation.
    trxs :
        The compiled transitions of the IPM.
    apply_matrix :
        A matrix defining how each event impacts each compartment
        (subtracting or adding individuals).
    events_leaving_compartment :
        A mapping from compartment index to the list of event indices which source from
        that compartment.
    source_compartment_for_event :
        A mapping from event index to the compartment index it sources from.
    """

    rume: RUME
    """The RUME."""

    data: DataResolver
    """The evaluated data attributes of the simulation."""

    trxs: list[CompiledTransition]
    """Compiled transitions."""

    apply_matrix: NDArray[SimDType]
    """
    A matrix defining how each event impacts each compartment
    (subtracting or adding individuals).
    """

    events_leaving_compartment: list[list[int]]
    """
    Mapping from compartment index to the list of event indices
    which source from that compartment.
    """

    source_compartment_for_event: list[int]
    """Mapping from event index to the compartment index it sources from."""

    @staticmethod
    def from_rume(rume: RUME, data: DataResolver) -> "IPMExecutor":
        ipm = rume.ipm
        csymbols = ipm.symbols.all_compartments
        return IPMExecutor(
            rume=rume,
            data=data,
            trxs=_compile_transitions(ipm),
            apply_matrix=_make_apply_matrix(ipm),
            # Calc list of events leaving each compartment (each may have 0, 1, or more)
            events_leaving_compartment=[
                [eidx for eidx, e in enumerate(ipm.events) if e.compartment_from == c]
                for c in csymbols
            ],
            # Calc the source compartment for each event
            source_compartment_for_event=[
                index_of(csymbols, e.compartment_from) for e in ipm.events
            ],
        )

    def stepper(
        self,
        clock: Sequence[Tick],
        world: World,
        save_tick_results: Callable[[int, SimArray, SimArray], None],
        rng: np.random.Generator,
    ) -> Callable[[], None]:
        """
        Get a function that will execute the IPM for a single tick when called,
        mutating the world state.

        Parameters
        ----------
        clock :
            The simulation clock, as a sequence of ticks.
        world :
            The world state.
        rng :
            The simulation RNG.
        """
        T, S, N, C, E = rume_dims(self.rume)

        attr_values_iter = self.data.resolve_txn_series(
            self.rume.ipm.requirements_dict.items(),
            self.rume.num_tau_steps,
        )

        tick_iter = iter(clock)

        # (home_node, visit_node, :)
        events = np.zeros((N, N, E), dtype=SimDType)
        compartments = np.zeros((N, N, C), dtype=SimDType)

        def step() -> None:
            tick, attrs = next(tick_iter), next(attr_values_iter)
            self._step(world, tick, attrs, rng, compartments, events)
            save_tick_results(tick.sim_index, compartments, events)

        return step

    def _step(
        self,
        world: World,
        tick: Tick,
        attr_values: list[AttributeValue],
        rng: np.random.Generator,
        out_compartments: SimArray,
        out_events: SimArray,
    ) -> None:
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
        T, S, N, C, E = rume_dims(self.rume)

        # TODO: optimization opportunity: re-use one copy of event/compartment arrays
        # (which would make this not parallel-safe)

        for visit_node in range(N):
            # Sum all cohorts present in this node to get an effective total population.
            cohorts = world.get_cohorts(visit_node)
            effective = reduce(
                lambda a, b: a + b,
                map(lambda x: x.compartments, cohorts),
            )

            # Determine how many events happen in this tick.
            node_events = self._events(
                tick, visit_node, effective, attr_values, rng
            )  # (E,) array

            # Distribute events to the cohorts, proportional to their population.
            cohort_events = self._distribute(cohorts, node_events, rng)  # (X,E) array

            # Now that events are assigned to cohorts,
            # convert to compartment deltas using apply matrix.
            world.apply_cohort_delta(
                visit_node,
                np.matmul(cohort_events, self.apply_matrix, dtype=SimDType),  # (X,C)
            )

            # Collect compartment/event info.
            for i, x in enumerate(cohorts):
                home = x.return_location
                out_events[home, visit_node, :] = cohort_events[i, :]
                out_compartments[home, visit_node, :] = x.compartments

    def _events(
        self,
        tick: Tick,
        node: int,
        effective_pop: SimArray,
        attr_values: list[AttributeValue],
        rng: np.random.Generator,
    ) -> SimArray:
        """
        Calculate how many events will happen this tick, correcting
        for the possibility of overruns. An (E,) array.
        """

        rate_args = [*effective_pop, *attr_values]

        # Evaluate the event rates and do random draws for all transition events.
        occur = np.zeros(self.rume.ipm.num_events, dtype=SimDType)
        index = 0
        for t in self.trxs:
            try:
                match t:
                    case CompiledEdge(rate_lambda):
                        # get rate from lambda expression, catch divide by zero error
                        rate = rate_lambda(rate_args)
                        if rate < 0:
                            err = self._get_default_error_args(rate_args, node, tick)
                            raise IPMSimLessThanZeroError(err)
                        occur[index] = rng.poisson(rate * tick.tau)
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
                        occur[index : (index + size)] = rng.multinomial(
                            n=rng.poisson(rate * tick.tau),
                            pvals=prob,
                        )
                index += t.size
            except (ZeroDivisionError, FloatingPointError):
                err = self._get_zero_division_args(rate_args, node, tick, t)
                raise IPMSimNaNError(err) from None

        # Check for event overruns leaving each compartment and correct counts.
        for cidx, eidxs in enumerate(self.events_leaving_compartment):
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
                    drawn0 = rng.hypergeometric(desired0, desired1, available)
                    occur[eidxs] = [drawn0, available - drawn0]
            else:
                # Compartment has more than two outwards edges:
                # use multivariate hypergeometric to select which events
                # "actually" happened.
                desired = occur[eidxs]
                if np.sum(desired) > available:
                    occur[eidxs] = rng.multivariate_hypergeometric(desired, available)
        return occur

    def _distribute(
        self,
        cohorts: Sequence[Cohort],
        events: NDArray[SimDType],  # (E,) array
        rng: np.random.Generator,
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
            cidx = self.source_compartment_for_event[eidx]
            if cidx == -1:
                # event is coming from an exogenous source
                # randomly distribute to cohorts based on their share of the population
                selected = rng.multinomial(
                    occur,
                    cohort_proportion,
                ).astype(SimDType)
            else:
                # event is coming from a modeled compartment
                selected = rng.multivariate_hypergeometric(
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
                [c.name.full for c in self.rume.ipm.compartments],
                rate_attrs[: self.rume.ipm.num_compartments],
            )
        }
        pvals = {
            attribute.name: value
            for attribute, value in zip(
                self.rume.ipm.requirements,
                rate_attrs[self.rume.ipm.num_compartments :],
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

        transition_index = self.trxs.index(transition)
        corr_transition = self.rume.ipm.transitions[transition_index]
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

        transition_index = self.trxs.index(transition)
        corr_transition = self.rume.ipm.transitions[transition_index]
        if isinstance(corr_transition, EdgeDef):
            trx = {str(corr_transition.name): corr_transition.rate}
            arg_list.append(("corresponding transition", trx))
        elif isinstance(corr_transition, ForkDef):
            trx = {str(corr_transition): corr_transition.rate}
            arg_list.append(("corresponding fork transition", trx))

        return arg_list


####################
# MovementExecutor #
####################


@dataclass(frozen=True)
class MovementExecutor:
    """
    The standard implementation of movement execution.

    rume :
        The RUME.
    data :
        The evaluated data attributes of the simulation.
    """

    rume: RUME
    """The RUME."""

    data: DataResolver
    """The evaluated data attributes of the simulation."""

    @staticmethod
    def from_rume(rume: RUME, data: DataResolver) -> "MovementExecutor":
        return MovementExecutor(rume=rume, data=data)

    def stepper(
        self,
        clock: Sequence[Tick],
        world: World,
        rng: np.random.Generator,
    ) -> Callable[[], None]:
        """
        Get a function that will execute movement for a single tick when called,
        mutating the world state.

        Parameters
        ----------
        clock :
            The simulation clock, as a sequence of ticks.
        world :
            The world state.
        rng :
            The simulation RNG.
        """
        tick_iter = iter(clock)

        clauses = []
        for strata, model in self.rume.mms.items():
            namespace = ModuleNamespace(gpm_strata(strata), "mm")
            for clause in model.clauses:
                ctx = Context.of(
                    namespace.to_absolute(clause.clause_name),
                    self.data,
                    self.rume.scope,
                    self.rume.time_frame,
                    self.rume.ipm,
                    np.random.default_rng(),
                )
                c = clause.with_context_internal(ctx)
                clauses.append((strata, c))

        def step() -> None:
            tick = next(tick_iter)
            self._step(world, tick, clauses, rng)

        return step

    def _step(
        self,
        world: World,
        tick: Tick,
        clauses: list[tuple[str, MovementClause]],
        rng: np.random.Generator,
    ) -> None:
        """
        Apply movement for this tick, mutating the world state.

        Parameters
        ----------
        tick :
            The tick to process.
        """
        # Clone and set context on clauses.
        clauses = []
        for strata, model in self.rume.mms.items():
            namespace = ModuleNamespace(gpm_strata(strata), "mm")
            for clause in model.clauses:
                ctx = Context.of(
                    namespace.to_absolute(clause.clause_name),
                    self.data,
                    self.rume.scope,
                    self.rume.time_frame,
                    self.rume.ipm,
                    rng,
                )
                c = clause.with_context_internal(ctx)
                clauses.append((strata, c))

        # _events.on_movement_start.publish(
        #     OnMovementStart(tick.sim_index, tick.day, tick.step)
        # )

        # Process travel clauses.
        total = 0
        for strata, clause in clauses:
            if not clause.is_active(tick):
                continue

            try:
                requested_movers = clause.evaluate(tick)
                np.fill_diagonal(requested_movers, 0)
            except Exception as e:
                # NOTE: catching exceptions here is necessary to get nice error messages
                # for some value error cause by incorrect parameter and/or clause
                # definition
                msg = (
                    f"Error from applying clause '{clause.__class__.__name__}': "
                    "see exception trace"
                )
                raise MMSimError(msg) from e

            available_movers = world.get_local_array()
            clause_event = calculate_travelers(
                clause.clause_name,
                self.rume.compartment_mobility[strata],
                requested_movers,
                available_movers,
                tick,
                rng,
            )
            # _events.on_movement_clause.publish(clause_event)
            travelers = clause_event.actual

            return_tick = resolve_tick_delta(
                self.rume.num_tau_steps,
                tick,
                clause.returns,
            )
            world.apply_travel(travelers, return_tick)
            total += travelers.sum()

        # Process return clause.
        return_movers_nnc = world.apply_return(tick, return_stats=True)
        return_movers_nn = return_movers_nnc.sum(axis=2)
        return_total = return_movers_nn.sum()
        total += return_total

        # _events.on_movement_clause.publish(
        #     OnMovementClause(
        #         tick.sim_index,
        #         tick.day,
        #         tick.step,
        #         "return",
        #         return_movers_nn,
        #         return_movers_nnc,
        #         return_total,
        #         False,
        #     )
        # )

        # _events.on_movement_finish.publish(
        #     OnMovementFinish(tick.sim_index, tick.day, tick.step, total)
        # )
