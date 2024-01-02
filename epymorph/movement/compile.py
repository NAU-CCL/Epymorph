"""
Compilation of movement models.
"""
from ast import FunctionDef
from functools import wraps
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from epymorph.code import ImmutableNamespace, compile_function, parse_function
from epymorph.error import AttributeException, MmCompileException, error_gate
from epymorph.movement.movement_model import (DynamicTravelClause,
                                              MovementContext,
                                              MovementFunction, MovementModel,
                                              TravelClause)
from epymorph.movement.parser import (ALL_DAYS, DailyClause, MovementClause,
                                      MovementSpec)
from epymorph.simulation import SimDType, Tick, TickDelta, epymorph_namespace


def compile_spec(ctx: MovementContext, spec: MovementSpec) -> MovementModel:
    """Compile a movement model from a spec, given a simulation context."""
    with error_gate("compiling the movement model", MmCompileException, AttributeException):

        # Prepare a namespace within which to execute our movement functions.
        global_namespace = _movement_global_namespace(ctx)

        # Execute predef (if any).
        if spec.predef is None:
            predef_result = {}
        else:
            predef_f = compile_function(
                parse_function(spec.predef.function),
                global_namespace
            )

            try:
                predef_result = predef_f()
            except KeyError as e:
                # NOTE: catching KeyError here will be necessary (to get nice error messages)
                # until we can properly validate the MM clauses.
                msg = f"Missing attribute {e} required by movement model predef."
                raise AttributeException(msg) from None

            if not isinstance(predef_result, dict):
                msg = f"Movement predef: did not return a dictionary result (got: {type(predef_result)})"
                raise MmCompileException(msg)

        # Merge predef into our namespace.
        global_namespace |= {'predef': predef_result}

        return MovementModel(
            tau_steps=spec.steps.step_lengths,
            clauses=[_compile_clause(c, global_namespace) for c in spec.clauses]
        )


def _movement_global_namespace(ctx: MovementContext) -> dict[str, Any]:
    """Make a safe namespace for user-defined movement functions."""
    def as_simdtype(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            result = func(*args, **kwargs)
            if np.isscalar(result):
                return SimDType(result)  # type: ignore
            else:
                return result.astype(SimDType)
        return wrapped_func

    global_namespace = epymorph_namespace()
    # Add rng functions to np namespace.
    np_ns = ImmutableNamespace({
        **global_namespace['np'].to_dict_shallow(),
        'poisson': as_simdtype(ctx.rng.poisson),
        'binomial': as_simdtype(ctx.rng.binomial),
        'multinomial': as_simdtype(ctx.rng.multinomial)
    })
    # Add simulation details.
    global_namespace |= {
        'geo': ctx.geo,
        'param': ctx.params,
        'nodes': ctx.dim.nodes,
        'np': np_ns,
    }
    return global_namespace


def _compile_clause(clause: MovementClause, global_namespace: dict[str, Any]) -> TravelClause:
    """Compiles a movement clause in a given namespace."""
    # Parse AST for the function.
    try:
        fn_ast = parse_function(clause.function)
        fn = compile_function(fn_ast, global_namespace)
    except Exception as e:
        msg = "Unable to parse and compile movement clause function."
        raise MmCompileException(msg) from e

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

            def returns(_ctx: MovementContext, _tick: Tick) -> TickDelta:
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
            raise MmCompileException(msg)
