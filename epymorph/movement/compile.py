"""
Compilation of movement models.
"""
import ast
from functools import wraps
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from epymorph.code import ImmutableNamespace, compile_function, parse_function
from epymorph.error import AttributeException, MmCompileException, error_gate
from epymorph.movement.movement_model import (DynamicTravelClause,
                                              MovementContext,
                                              MovementFunction, MovementModel,
                                              PredefParams, TravelClause)
from epymorph.movement.parser import (ALL_DAYS, DailyClause, MovementClause,
                                      MovementSpec)
from epymorph.simulation import SimDType, Tick, TickDelta, epymorph_namespace


def _empty_predef(_ctx: MovementContext) -> PredefParams:
    """A placeholder predef function for when none is given by the movement spec."""
    return {}


def compile_spec(ctx: MovementContext, spec: MovementSpec) -> MovementModel:
    """Compile a movement model from a spec, given a simulation context."""
    with error_gate("compiling the movement model", MmCompileException, AttributeException):
        # Prepare a namespace within which to execute our movement functions.
        global_namespace = _movement_global_namespace(ctx)

        # Compile predef (if any).
        if spec.predef is None:
            predef_f = _empty_predef
        else:
            orig_ast = parse_function(spec.predef.function)
            trns_ast = transform_predef_ast(orig_ast)
            predef_f = compile_function(trns_ast, global_namespace)

        def predef_context_hash(ctx: MovementContext) -> int:
            # NOTE: This is a placeholder predef hash function
            # that will recalculate the predef if any change is made to the context.
            # Fine for now, but we could go finer-grained than that
            # and only recalc if something changes that the predef code
            # actually uses. For this we'll have to extract references
            # from the predef AST.
            return hash(ctx.version)

        return MovementModel(
            tau_steps=spec.steps.step_lengths,
            predef=predef_f,
            predef_context_hash=predef_context_hash,
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
        'MovementContext': MovementContext,
        'PredefParams': PredefParams,
        'np': np_ns,
    }
    return global_namespace


class MovementFunctionTransformer(ast.NodeTransformer):
    """
    Transforms movement clause code so that we can pass context, etc.,
    via function arguments instead of the namespace. The goal is to
    simplify the function interface for end users while still maintaining
    good performance characteristics when parameters change during
    a simulation run (i.e., not have to recompile the functions every time
    the params change).

    A function like:

    def commuters(t):
        typical = np.minimum(
            geo['population'][:],
            predef['commuters_by_node'],
        )
        actual = np.binomial(typical, param['move_control'])
        return np.multinomial(actual, predef['commuting_probability'])

    Will be rewritten as:

    def commuters(ctx, predef, t):
        typical = np.minimum(
            ctx.geo['population'][:],
            predef['commuters_by_node'],
        )
        actual = np.binomial(typical, ctx.param['move_control'])
        return np.multinomial(actual, predef['commuting_probability'])

    If `is_predef` is given as true, we'll only add context to the args,
    not predef (makes sense).
    """

    is_predef: bool

    def __init__(self, is_predef: bool):
        self.is_predef = is_predef

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        """Modify references to dictionaries that should be in context."""
        if isinstance(node.value, ast.Name):
            if node.value.id in ['geo', 'params']:
                node.value = ast.Attribute(
                    value=ast.Name(id='ctx', ctx=ast.Load()),
                    attr=node.value.id,
                    ctx=ast.Load(),
                )
                return node
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        """Modify function parameters."""
        new_node = self.generic_visit(node)
        if isinstance(new_node, ast.FunctionDef):
            ctx_arg = ast.arg(
                arg='ctx',
                annotation=ast.Name(id='MovementContext', ctx=ast.Load()),
            )
            predef_arg = ast.arg(
                arg='predef',
                annotation=ast.Name(id='PredefParams', ctx=ast.Load()),
            )
            if self.is_predef:
                args = [ctx_arg, *new_node.args.args]
            else:
                args = [ctx_arg, predef_arg, *new_node.args.args]
            new_node.args.args = args
        return new_node


def transform_movement_ast(orig_ast: ast.FunctionDef) -> ast.FunctionDef:
    """Transforms movement clause code. See MovementFunctionTransformer for details."""
    trns_ast = MovementFunctionTransformer(is_predef=False).visit(orig_ast)
    ast.fix_missing_locations(trns_ast)
    return trns_ast


def transform_predef_ast(orig_ast: ast.FunctionDef) -> ast.FunctionDef:
    """Transforms movement predef code. See MovementFunctionTransformer for details."""
    trns_ast = MovementFunctionTransformer(is_predef=True).visit(orig_ast)
    ast.fix_missing_locations(trns_ast)
    return trns_ast


def _compile_clause(clause: MovementClause, global_namespace: dict[str, Any]) -> TravelClause:
    """Compiles a movement clause in a given namespace."""
    # Parse AST for the function.
    try:
        fn_ast = transform_movement_ast(parse_function(clause.function))
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


def _adapt_move_function(fn: Callable, fn_ast: ast.FunctionDef) -> MovementFunction:
    """
    Wrap the user-provided function in order to handle functions of different arity.
    Movement functions as specified by the user can have signature:
    f(tick); f(tick, src); or f(tick, src, dst).
    """
    match len(fn_ast.args.args):
        # Remember `fn` has been transformed, so if the user gave 1 arg we added 2 for a total of 3.
        case 3:
            @wraps(fn)
            def fn_arity1(ctx: MovementContext, predef: PredefParams, tick: Tick) -> NDArray[SimDType]:
                requested = fn(ctx, predef, tick)
                np.fill_diagonal(requested, 0)
                return requested
            return fn_arity1

        case 4:
            @wraps(fn)
            def fn_arity2(ctx: MovementContext, predef: PredefParams, tick: Tick) -> NDArray[SimDType]:
                N = ctx.dim.nodes
                requested = np.zeros((N, N), dtype=SimDType)
                for n in range(N):
                    requested[n, :] = fn(ctx, predef, tick, n)
                np.fill_diagonal(requested, 0)
                return requested
            return fn_arity2

        case 5:
            @wraps(fn)
            def fn_arity3(ctx: MovementContext, predef: PredefParams, tick: Tick) -> NDArray[SimDType]:
                N = ctx.dim.nodes
                requested = np.zeros((N, N), dtype=SimDType)
                for i, j in np.ndindex(N, N):
                    requested[i, j] = fn(ctx, predef, tick, i, j)
                np.fill_diagonal(requested, 0)
                return requested
            return fn_arity3

        case invalid_num_args:
            msg = f"Movement clause '{fn_ast.name}' has an invalid number of arguments ({invalid_num_args})"
            raise MmCompileException(msg)
