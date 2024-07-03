"""
Compilation of movement models.
"""
import ast
from functools import wraps
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from epymorph.code import (ImmutableNamespace, compile_function,
                           epymorph_namespace, parse_function)
from epymorph.data_type import SimDType
from epymorph.error import AttributeException, MmCompileException, error_gate
from epymorph.movement.movement_model import (DynamicTravelClause,
                                              MovementContext,
                                              MovementFunction, MovementModel,
                                              PredefData, TravelClause)
from epymorph.movement.parser import (ALL_DAYS, DailyClause, MovementClause,
                                      MovementSpec)
from epymorph.simulation import AttributeDef, Tick, TickDelta
from epymorph.util import identity


def _empty_predef(_ctx: MovementContext) -> PredefData:
    """A placeholder predef function for when none is given by the movement spec."""
    return {}


def compile_spec(
    spec: MovementSpec,
    rng: np.random.Generator,
    name_override: Callable[[str], str] = identity,
) -> MovementModel:
    """
    Compile a movement model from a spec. Requires a reference to the random number generator
    that will be used to execute the movement model.
    By default, clauses will be given a name from the spec file, but you can override
    that naming behavior by providing the `name_override` function.
    """
    with error_gate("compiling the movement model", MmCompileException, AttributeException):
        # Prepare a namespace within which to execute our movement functions.
        global_namespace = _movement_global_namespace(rng)

        # Compile predef (if any).
        if spec.predef is None:
            predef_f = _empty_predef
        else:
            orig_ast = parse_function(spec.predef.function)
            transformer = PredefFunctionTransformer(spec.attributes)
            trns_ast = transformer.visit_and_fix(orig_ast)
            predef_f = compile_function(trns_ast, global_namespace)

        return MovementModel(
            tau_steps=spec.steps.step_lengths,
            attributes=spec.attributes,
            predef=predef_f,
            clauses=[_compile_clause(c, spec.attributes, global_namespace, name_override)
                     for c in spec.clauses]
        )


def _movement_global_namespace(rng: np.random.Generator) -> dict[str, Any]:
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

    global_namespace = epymorph_namespace(SimDType)
    # Add rng functions to np namespace.
    np_ns = ImmutableNamespace({
        **global_namespace['np'].to_dict_shallow(),
        'poisson': as_simdtype(rng.poisson),
        'binomial': as_simdtype(rng.binomial),
        'multinomial': as_simdtype(rng.multinomial)
    })
    # Add simulation details.
    global_namespace |= {
        'MovementContext': MovementContext,
        'PredefData': PredefData,
        'np': np_ns,
    }
    return global_namespace


def _compile_clause(
    clause: MovementClause,
    model_attributes: Sequence[AttributeDef],
    global_namespace: dict[str, Any],
    name_override: Callable[[str], str] = identity,
) -> TravelClause:
    """Compiles a movement clause in a given namespace."""
    # Parse AST for the function.
    try:
        orig_ast = parse_function(clause.function)
        transformer = ClauseFunctionTransformer(model_attributes)
        fn_ast = transformer.visit_and_fix(orig_ast)
        fn = compile_function(fn_ast, global_namespace)
    except MmCompileException as e:
        raise e
    except Exception as e:
        msg = "Unable to parse and compile movement clause function."
        raise MmCompileException(msg) from e

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
                name=name_override(fn_ast.name),
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
        # Remember `fn` has been transformed, so if the user gave 1 arg we added 1 for a total of 2.
        case 2:
            @wraps(fn)
            def fn_arity1(ctx: MovementContext, tick: Tick) -> NDArray[SimDType]:
                requested = fn(ctx, tick)
                np.fill_diagonal(requested, 0)
                return requested
            return fn_arity1

        case 3:
            @wraps(fn)
            def fn_arity2(ctx: MovementContext, tick: Tick) -> NDArray[SimDType]:
                N = ctx.dim.nodes
                requested = np.zeros((N, N), dtype=SimDType)
                for n in range(N):
                    requested[n, :] = fn(ctx, tick, n)
                np.fill_diagonal(requested, 0)
                return requested
            return fn_arity2

        case 4:
            @wraps(fn)
            def fn_arity3(ctx: MovementContext, tick: Tick) -> NDArray[SimDType]:
                N = ctx.dim.nodes
                requested = np.zeros((N, N), dtype=SimDType)
                for i, j in np.ndindex(N, N):
                    requested[i, j] = fn(ctx, tick, i, j)
                np.fill_diagonal(requested, 0)
                return requested
            return fn_arity3

        case invalid_num_args:
            msg = f"Movement clause '{fn_ast.name}' has an invalid number of arguments ({invalid_num_args})"
            raise MmCompileException(msg)


# Code transformers

class HasLineNo(Protocol):
    lineno: int


class _MovementCodeTransformer(ast.NodeTransformer):
    """
    This class defines the logic that can be shared between Predef and Clause function
    transformers. Some functionality might be more than is technically necessary for either
    case, but only if that extra functionality is effectively harmless.
    """

    check_attributes: bool
    attributes: Mapping[str, AttributeDef]

    def __init__(self, attributes: Sequence[AttributeDef]):
        # NOTE: for the sake of backwards compatibility, MovementModel attribute declarations
        # are optional; so our approach will be that attributes will only be checked if at least
        # one attribute declaration is provided.
        if len(attributes) == 0:
            self.check_attributes = False
            self.attributes = {}
        else:
            self.check_attributes = True
            self.attributes = {a.name: a for a in attributes}

    def _report_line(self, node: HasLineNo):
        return f"Line: {node.lineno}"

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        """Modify references to data and predef pseudo-dictionaries."""

        if isinstance(node.value, ast.Name) and isinstance(node.slice, ast.Constant) and node.value.id in ['data', 'predef']:
            source = node.value.id
            attr_name = node.slice.value

            # Check data attributes against declarations (but ignore predefs).
            if self.check_attributes and source == 'data' and attr_name not in self.attributes:
                msg = f"Movement model is using an undeclared attribute: `data[{attr_name}]`. "\
                    f"Please add a suitable attribute declaration. ({self._report_line(node)})"
                raise MmCompileException(msg)

            # NOTE: what we are *NOT* doing is checking if usage of predef attributes are
            # actually provided by the predef function. Doing this at compile time would be
            # exceedingly difficult, as we'd have to scrape and analyze all code that contributes to
            # the returned dictionary's keys. In simple cases this might be straight-forward, but not
            # in the general case. For the time being, this will remain a simulation-time error.

            # Rewrite to access via context resolver.
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Attribute(
                        value=ast.Name(id='ctx', ctx=ast.Load()),
                        attr='data',
                        ctx=ast.Load(),
                    ),
                    attr='resolve_name',
                    ctx=ast.Load(),
                ),
                args=[node.slice],
                keywords=[],
            )

        return self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        """Modify references to objects that should be in context."""
        if isinstance(node.value, ast.Name) and node.value.id in ['dim']:
            node.value = ast.Attribute(
                value=ast.Name(id='ctx', ctx=ast.Load()),
                attr=node.value.id,
                ctx=ast.Load(),
            )
            return node
        return self.generic_visit(node)

    def visit_and_fix(self, node: ast.AST) -> Any:
        """
        Shortcut for visiting the node and then running
        ast.fix_missing_locations() on the result before returning it.
        """
        transformed = self.visit(node)
        ast.fix_missing_locations(transformed)
        return transformed


class PredefFunctionTransformer(_MovementCodeTransformer):
    """
    Transforms movement model predef code. This is the dual of
    ClauseFunctionTransformer (below; see that for additional description),
    but specialized for predef which is similar but slightly different.
    Most importantly, this transforms the function signature to have the context
    as the first parameter.
    """

    def _report_line(self, node: HasLineNo):
        return f"predef line: {node.lineno}"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        """Modify function parameters."""
        new_node = self.generic_visit(node)
        if isinstance(new_node, ast.FunctionDef):
            ctx_arg = ast.arg(
                arg='ctx',
                annotation=ast.Name(id='MovementContext', ctx=ast.Load()),
            )
            new_node.args.args = [ctx_arg, *new_node.args.args]
        return new_node


class ClauseFunctionTransformer(_MovementCodeTransformer):
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
            data['population'][:],
            data['commuters_by_node'],
        )
        actual = np.binomial(typical, data['move_control'])
        return np.multinomial(actual, predef['commuting_probability'])

    Will be rewritten as:

    def commuters(ctx, t):
        typical = np.minimum(
            ctx.data.resolve_name('population')[:],
            ctx.data.resolve_name('commuters_by_node'),
        )
        actual = np.binomial(typical, ctx.data.resolve_name('move_control'))
        return np.multinomial(actual, ctx.data.resolve_name('commuting_probability'))
    """

    clause_name: str = "<unknown clause>"

    def _report_line(self, node: HasLineNo):
        return f"{self.clause_name} line: {node.lineno}"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        """Modify function parameters."""
        self.clause_name = f"`{node.name}`"
        new_node = self.generic_visit(node)
        if isinstance(new_node, ast.FunctionDef):
            ctx_arg = ast.arg(
                arg='ctx',
                annotation=ast.Name(id='MovementContext', ctx=ast.Load()),
            )
            new_node.args.args = [ctx_arg, *new_node.args.args]
        return new_node
