"""
The basis of the movement model system in epymorph.
This module contains all of the elements needed to define a
movement model, but execution of it is left to the mm_exec module.
"""
from abc import ABC, abstractmethod
from ast import FunctionDef
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Protocol

import numpy as np
from numpy.typing import NDArray

from epymorph.compartment_model import CompartmentModel
from epymorph.engine.context import make_namespace
from epymorph.error import MmValidationException
from epymorph.geo.geo import Geo
from epymorph.parser.move_clause import ALL_DAYS, DailyClause, MovementClause
from epymorph.parser.movement import MovementSpec
from epymorph.simulation import (Params, SimDimensions, SimDType, Tick,
                                 TickDelta, base_namespace)
from epymorph.util import ImmutableNamespace, compile_function, parse_function


class MovementContext(Protocol):
    """The subset of the ExecutionContext that the movement model clauses need."""
    # This machine avoids circular deps.
    dim: SimDimensions
    geo: Geo
    ipm: CompartmentModel
    params: Params
    rng: np.random.Generator


class TravelClause(ABC):
    """A clause moving individuals from their home location to another."""

    name: str

    @abstractmethod
    def mask(self, ctx: MovementContext) -> NDArray[np.bool_]:
        """Calculate the movement mask for this clause."""

    @abstractmethod
    def predicate(self, ctx: MovementContext, tick: Tick) -> bool:
        """Should this clause apply this tick?"""

    @abstractmethod
    def requested(self, ctx: MovementContext, tick: Tick) -> NDArray[SimDType]:
        """Evaluate this clause for the given tick, returning a requested movers array (N,N)."""

    @abstractmethod
    def returns(self, ctx: MovementContext, tick: Tick) -> TickDelta:
        """Calculate when this clause's movers should return (which may vary from tick-to-tick)."""


MaskPredicate = Callable[[MovementContext], NDArray[np.bool_]]
"""
A predicate which creates a per-IPM-compartment mask:
should this compartment be subject to movement by this clause?
"""

MovementPredicate = Callable[[MovementContext, Tick], bool]
"""A predicate which decides if a clause should fire this tick."""

MovementFunction = Callable[[MovementContext, Tick], NDArray[SimDType]]
"""
A function which calculates the requested number of individuals to move due to this clause this tick.
Returns an (N,N) array of integers.
"""

ReturnsFunction = Callable[[MovementContext, Tick], TickDelta]
"""A function which decides when this clause's movers should return."""


class DynamicTravelClause(TravelClause):
    """
    A travel clause implementation where each method proxies to a lambda.
    This allows us to build travel clauses dynamically at runtime.
    """

    name: str

    _mask: MaskPredicate
    _move: MovementPredicate
    _requested: MovementFunction
    _returns: ReturnsFunction

    def __init__(self,
                 name: str,
                 mask_predicate: MaskPredicate,
                 move_predicate: MovementPredicate,
                 requested: MovementFunction,
                 returns: ReturnsFunction):
        self.name = name
        self._mask = mask_predicate
        self._move = move_predicate
        self._requested = requested
        self._returns = returns

    def mask(self, ctx: MovementContext) -> NDArray[np.bool_]:
        return self._mask(ctx)

    def predicate(self, ctx: MovementContext, tick: Tick) -> bool:
        return self._move(ctx, tick)

    def requested(self, ctx: MovementContext, tick: Tick) -> NDArray[SimDType]:
        return self._requested(ctx, tick)

    def returns(self, ctx: MovementContext, tick: Tick) -> TickDelta:
        return self._returns(ctx, tick)


@dataclass(frozen=True)
class MovementModel:
    """
    The movement model divides a day into simulation parts (tau steps) under the assumption
    that each day part will have movement characteristics relevant to the simulation.
    That is: there is no reason to have tau steps smaller than 1 day unless it's relevant
    to movement.
    """

    tau_steps: list[float]
    """The tau steps for the simulation."""
    clauses: list[TravelClause]
    """The clauses which express the movement model"""


############################################################
# MovementModel compilation
############################################################


def make_global_namespace(ctx: MovementContext) -> dict[str, Any]:
    """Make a safe namespace for user-defined functions."""
    def as_simdtype(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            result = func(*args, **kwargs)
            if np.isscalar(result):
                return SimDType(result)  # type: ignore
            else:
                return result.astype(SimDType)
        return wrapped_func

    global_namespace = base_namespace()
    # Add rng functions to np namespace.
    np_ns = ImmutableNamespace({
        **global_namespace['ns'].to_dict_shallow(),
        'poisson': as_simdtype(ctx.rng.poisson),
        'binomial': as_simdtype(ctx.rng.binomial),
        'multinomial': as_simdtype(ctx.rng.multinomial)
    })
    # Add simulation details.
    global_namespace |= {
        'geo': ctx.geo,
        'params': ctx.params,
        'nodes': ctx.dim.nodes,
        'np': np_ns,
    }
    return global_namespace


def compile_spec(ctx: MovementContext, spec: MovementSpec) -> MovementModel:
    """Compile a movement model from a spec, given a simulation context."""
    try:
        # Prepare a namespace within which to execute our movement functions.
        global_namespace = make_global_namespace(ctx)

        # Execute predef (if any).
        if spec.predef is None:
            predef_result = {}
        else:
            predef_f = compile_function(
                parse_function(spec.predef.function),
                global_namespace
            )
            predef_result = predef_f()
            if not isinstance(predef_result, dict):
                msg = f"Movement predef: did not return a dictionary result (got: {type(predef_result)})"
                raise MmValidationException(msg)

        # Merge predef into our namespace.
        global_namespace |= {'predef': predef_result}

        def compile_clause(clause: MovementClause) -> TravelClause:
            """Compiles a movement clause in this context."""
            # Parse AST for the function.
            try:
                fn_ast = parse_function(clause.function)
                fn = compile_function(fn_ast, global_namespace)
            except Exception as e:
                msg = "Unable to compile movement clause function."
                raise MmValidationException(msg) from e

            # Construct a mask for IPM compartments subject to movement.
            def mask_predicate(ctx: MovementContext) -> NDArray[np.bool_]:
                return np.array(
                    ['immobile' not in c.tags for c in ctx.ipm.compartments],
                    dtype=np.bool_
                )

            # Handle different types of MovementClause.
            match clause:
                case DailyClause():
                    clause_weekdays = set(
                        i for (i, d) in enumerate(ALL_DAYS)
                        if d in clause.days
                    )

                    def move_predicate(_ctx: MovementContext, tick: Tick) -> bool:
                        return clause.leave_step == tick.step and \
                            tick.date.weekday() in clause_weekdays

                    def returns(_ctx, _tick) -> TickDelta:
                        return TickDelta(
                            days=clause.duration.to_days(),
                            step=clause.return_step
                        )

                    return DynamicTravelClause(
                        name=fn_ast.name,
                        mask_predicate=mask_predicate,
                        move_predicate=move_predicate,
                        requested=_adapt_move_function(fn, fn_ast),
                        returns=returns
                    )

        return MovementModel(
            tau_steps=spec.steps.step_lengths,
            clauses=[compile_clause(c) for c in spec.clauses]
        )

    except MmValidationException as e:
        raise e
    except Exception as e:
        msg = "Unknown exception during movement model compilation."
        raise MmValidationException(msg) from e


def _adapt_move_function(fn: Callable, fn_ast: FunctionDef) -> MovementFunction:
    """
    Wrap the user-provided function in order to handle functions of different arity.
    Movement functions have signature: f(tick); f(tick, src); or f(tick, src, dst).
    """
    match len(fn_ast.args.args):
        case 1:
            @wraps(fn)
            def fn_arity1(_ctx: MovementContext, tick: Tick) -> NDArray[SimDType]:
                requested = fn(tick)
                np.fill_diagonal(requested, 0)
                return requested
            return fn_arity1

        case 2:
            @wraps(fn)
            def fn_arity2(ctx: MovementContext, tick: Tick) -> NDArray[SimDType]:
                N = ctx.dim.nodes
                requested = np.zeros((N, N), dtype=SimDType)
                for n in range(N):
                    requested[n, :] = fn(tick, n)
                np.fill_diagonal(requested, 0)
                return requested
            return fn_arity2

        case 3:
            @wraps(fn)
            def fn_arity3(ctx: MovementContext, tick: Tick) -> NDArray[SimDType]:
                N = ctx.dim.nodes
                requested = np.zeros((N, N), dtype=SimDType)
                for i, j in np.ndindex(N, N):
                    requested[i, j] = fn(tick, i, j)
                np.fill_diagonal(requested, 0)
                return requested
            return fn_arity3

        case invalid_num_args:
            msg = f"Movement clause '{fn_ast.name}' has an invalid number of arguments ({invalid_num_args})"
            raise MmValidationException(msg)
