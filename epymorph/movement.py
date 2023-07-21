from __future__ import annotations

from typing import Any, Callable, NamedTuple

import numpy as np

from epymorph.clock import TickDelta
from epymorph.context import SimContext
from epymorph.movement_clause import (Clause, GeneralClause, Predicates,
                                      Return, Sequence)
from epymorph.parser.move_clause import Daily
from epymorph.parser.move_predef import Predef
from epymorph.parser.movement import MovementSpec, movement_spec
from epymorph.util import compile_function, parse_function


class MovementBuilder:

    taus: list[np.double]
    clause_compiler: Callable[[SimContext], Clause]

    def __init__(self, taus: list[np.double], clause_compiler: Callable[[SimContext], Clause]):
        assert len(taus) > 0, "Must supply at least one tau step."
        assert np.sum(taus) == np.double(1), "Tau steps must sum to 1."
        self.taus = taus
        self.clause_compiler = clause_compiler

    def verify(self, ctx: SimContext) -> None:
        # TODO: how do we verify the context for movement?
        pass

    def build(self, ctx: SimContext) -> Movement:
        return Movement(self.taus, self.clause_compiler(ctx))


class Movement(NamedTuple):
    """
    The movement model divides a day into simulation parts (tau steps) under the assumption
    that each day part will have movement characteristics relevant to the simulation.
    That is: there is no reason to have tau steps smaller than 1 day unless it's relevant to movement.
    """
    taus: list[np.double]
    """The tau steps for the simulation."""
    clause: Clause
    """A clause which expresses the movement model (most likely as a Sequence clause which is a combination of other, conditional clauses.)"""


def parse_clause(clause_spec: Daily) -> Callable[[SimContext, dict], Clause]:
    """Parse a clause specification yielding a function capable of compiling it into a reified Clause."""
    try:
        f_def = parse_function(clause_spec.f)
    except:
        raise Exception(f"Movement clause: not a valid function")

    prd = Predicates.daylist(days=clause_spec.days,
                             step=clause_spec.leave_step - 1)
    ret = TickDelta(days=clause_spec.duration.to_days(),
                    step=clause_spec.return_step - 1)

    def ctp(tags: list[str]) -> bool:
        return 'immobile' not in tags

    num_args = len(f_def.args.args)
    if num_args == 2:
        f_shape = GeneralClause.by_row
    elif num_args == 3:
        f_shape = GeneralClause.by_cross
    else:
        raise Exception(
            f"Movement clause: invalid number of arguments ({num_args})")

    def compile_clause(ctx: SimContext, global_namespace: dict) -> Clause:
        f = compile_function(f_def, global_namespace)
        return f_shape(ctx, f_def.name, prd, ret, f, ctp)

    return compile_clause


def _execute_predef(predef: Predef, global_namespace: dict) -> dict:
    """Compile and execute the predef section of a movement spec, yielding its return value."""
    predef_f = compile_function(parse_function(predef.f), global_namespace)
    result = predef_f()
    if not isinstance(result, dict):
        raise Exception(
            f"Movement predef: did not return a dictionary result (got: {type(result)})")
    return result


def _make_global_namespace(ctx: SimContext) -> dict[str, Any]:
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


def load_movement_spec(spec_string: str) -> MovementBuilder:
    results = movement_spec.parse_string(spec_string, parse_all=True)
    spec: MovementSpec = results[0]  # type: ignore

    clause_compilers = [parse_clause(c) for c in spec.clauses]

    def compile_clause(ctx: SimContext) -> Clause:
        global_namespace = _make_global_namespace(ctx)
        # t0 = time.perf_counter()
        predef = {} if spec.predef is None else _execute_predef(
            spec.predef, global_namespace)
        # t1 = time.perf_counter()
        # print(f"Executed predef in {(1000 * (t1 - t0)):.3f} ms")
        global_namespace = global_namespace | {'predef': predef}
        clauses = [cc(ctx, global_namespace)
                   for cc in clause_compilers]
        clauses.append(Return(ctx))
        return Sequence(clauses)

    taus = [np.double(x) for x in spec.steps.steps]
    return MovementBuilder(taus, compile_clause)


def check_movement_spec(spec_string: str) -> None:
    movement_spec.parse_string(spec_string, parse_all=True)
    # If no Exceptions are thrown, it's good.
    # TODO: need to do some thinking about Exception crafting
    # to produce the most helpful error messaging here.
