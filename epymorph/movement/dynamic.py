from __future__ import annotations

from typing import Any, Callable

import numpy as np
from attr import dataclass
from numpy.typing import NDArray

from epymorph.clock import Tick, TickDelta
from epymorph.context import Compartments, SimContext
from epymorph.movement.clause import (RETURN, ArrayClause, Clause,
                                      CompartmentPredicate, RowClause,
                                      TravelClause)
from epymorph.movement.engine import Movement, MovementBuilder
from epymorph.parser.move_clause import ALL_DAYS, Daily, DayOfWeek
from epymorph.parser.move_predef import Predef
from epymorph.parser.movement import MovementSpec, movement_spec
from epymorph.util import compile_function, parse_function


@dataclass
class DynamicClauseInfo:
    name: str
    returns: TickDelta
    # Predicate info
    prd_step: int
    prd_days: list[DayOfWeek]
    # Compartment mask info
    ctp: CompartmentPredicate
    tags: list[list[str]]


class DynamicTravelClause(TravelClause):
    # Predicate: step and weekdays
    prd_step: int
    prd_days: set[int]

    def __init__(self, info: DynamicClauseInfo):
        self.name = info.name
        self.returns = info.returns
        # Compute movement mask.
        self.movement_mask = np.array(
            [info.ctp(ts) for ts in info.tags], dtype=bool)
        # Prepare predicate
        self.prd_step = info.prd_step
        self.prd_days = set(i for (i, d) in enumerate(ALL_DAYS)
                            if d in info.prd_days)

    def predicate(self, tick: Tick) -> bool:
        return self.prd_step == tick.step and \
            tick.date.weekday() in self.prd_days


class DynamicArrayClause(ArrayClause, DynamicTravelClause):
    f: Callable[[Tick], Compartments]

    def __init__(self, info: DynamicClauseInfo, f: Callable[[Tick], Compartments]):
        super().__init__(info)
        self.f = f

    def apply(self, tick: Tick) -> Compartments:
        return self.f(tick)


class DynamicRowClause(RowClause, DynamicTravelClause):
    f: Callable[[Tick, int], Compartments]

    def __init__(self, info: DynamicClauseInfo, f: Callable[[Tick, int], Compartments]):
        super().__init__(info)
        self.f = f

    def apply(self, tick: Tick, src_index: int) -> Compartments:
        return self.f(tick, src_index)


class DynamicCellClause(RowClause, DynamicTravelClause):
    f: Callable[[Tick, int, int], Compartments]

    def __init__(self, info: DynamicClauseInfo, f: Callable[[Tick, int, int], Compartments]):
        super().__init__(info)
        self.f = f

    def apply(self, tick: Tick, src_index: int, dst_index: int) -> Compartments:
        return self.f(tick, src_index, dst_index)


Namespace = dict[str, Any]
ClauseCompiler = Callable[[SimContext, Namespace], Clause]


def to_clause_compiler(clause_spec: Daily) -> ClauseCompiler:
    """Parse a clause specification yielding a function capable of compiling it into a reified Clause."""

    # Get AST for function
    try:
        f_def = parse_function(clause_spec.f)
    except:
        raise Exception("Movement clause: not a valid function")

    num_args = len(f_def.args.args)
    if num_args not in [1, 2, 3]:
        raise Exception(
            f"Movement clause: invalid number of arguments ({num_args})")

    def compile_clause(ctx: SimContext, global_namespace: Namespace) -> Clause:
        clause_info = DynamicClauseInfo(
            name=f_def.name,
            returns=TickDelta(days=clause_spec.duration.to_days(),
                              step=clause_spec.return_step - 1),
            # clause predicate info (for the only form currently supported)
            prd_step=clause_spec.leave_step - 1,
            prd_days=clause_spec.days,
            # default/placeholder CTP
            ctp=lambda tags: 'immobile' not in tags,
            tags=ctx.compartment_tags
        )
        f = compile_function(f_def, global_namespace)
        match num_args:
            case 1:
                return DynamicArrayClause(clause_info, f)
            case 2:
                return DynamicRowClause(clause_info, f)
            case 3:
                return DynamicCellClause(clause_info, f)
            case _:
                # We should catch this case above, but just in case...
                raise Exception(
                    f"Movement clause: invalid number of arguments ({num_args})")

    return compile_clause


def execute_predef(predef: Predef, global_namespace: Namespace) -> Namespace:
    """Compile and execute the predef section of a movement spec, yielding its return value."""
    predef_f = compile_function(parse_function(predef.f), global_namespace)
    result = predef_f()
    if not isinstance(result, dict):
        raise Exception(
            f"Movement predef: did not return a dictionary result (got: {type(result)})")
    return result


def make_global_namespace(ctx: SimContext) -> Namespace:
    """Make a safe namespace for user-defined functions."""
    return {
        # simulation data
        'geo': ctx.geo,
        'nodes': ctx.nodes,
        'param': ctx.param,
        # rng functions
        'poisson': ctx.rng.poisson,
        'binomial': ctx.rng.binomial,
        'multinomial': ctx.rng.multinomial,
        # numpy utility functions
        'array': np.array,
        'zeros': np.zeros,
        'zeros_like': np.zeros_like,
        'newaxis': np.newaxis,
        'exp': np.exp,
        'radians': np.radians,
        'sin': np.sin,
        'cos': np.cos,
        'arcsin': np.arcsin,
        'arctan2': np.arctan2,
        'sqrt': np.sqrt,
        'subtract': np.subtract,
        'multiply': np.multiply,
        'divide': np.divide,
        # restricted functions
        # TODO: there are probably more restrictions to add
        # TODO: in fact, this is probably not sufficient as a security model,
        # though it'll do for now
        'breakpoint': None,
        'compile': None,
        'eval': None,
        'exec': None,
        'globals': None,
        'print': None
    }


class DynamicMovementBuilder(MovementBuilder):
    taus: list[float]
    spec: MovementSpec

    compilers: list[ClauseCompiler] | None
    namespace: Namespace | None

    def __init__(self, spec: MovementSpec):
        taus = spec.steps.steps
        assert len(taus) > 0, "Must supply at least one tau step."
        assert sum(taus) == 1, "Tau steps must sum to 1."
        self.taus = taus
        self.spec = spec
        self.compilers = None
        self.namespace = None

    def verify(self, ctx: SimContext) -> None:
        # TODO: how do we verify the context for movement?
        pass

    def build(self, ctx: SimContext) -> Movement:
        # results = movement_spec.parse_string(spec_string, parse_all=True)
        # spec: MovementSpec = results[0]  # type: ignore

        if self.namespace is None:
            ns = make_global_namespace(ctx)

            # t0 = time.perf_counter()
            predef = {} if self.spec.predef is None else \
                execute_predef(self.spec.predef, ns)
            # t1 = time.perf_counter()
            # print(f"Executed predef in {(1000 * (t1 - t0)):.3f} ms")

            ns = ns | {'predef': predef}
            self.namespace = ns

        if self.compilers is None:
            self.compilers = [to_clause_compiler(c) for c in self.spec.clauses]

        # WARNING: We're only making a shallow copy of the namespace for each run,
        # so if a movement function modifies the underlying data, the
        # modification will remain for subsequent runs.
        ns = self.namespace.copy()

        clauses = [
            *(compile(ctx, ns) for compile in self.compilers),
            RETURN
        ]

        return Movement(self.spec.steps.steps, clauses)


def check_movement_spec(spec_string: str) -> None:
    movement_spec.parse_string(spec_string, parse_all=True)
    # If no Exceptions are thrown, it's good.
    # TODO: need to do some thinking about Exception crafting
    # to produce the most helpful error messaging here.
