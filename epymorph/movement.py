from __future__ import annotations

import ast
from typing import Any, Callable, NamedTuple

import numpy as np

from epymorph.clock import TickDelta
from epymorph.context import SimContext
from epymorph.movement_clause import (Clause, GeneralClause, Predicates,
                                      Return, Sequence)
from epymorph.parser.move_clause import Daily
from epymorph.parser.movement import MovementSpec, movement_spec


class MovementBuilder:
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
    f_ast = ast.parse(clause_spec.f, '<string>', mode='exec')
    f_def = f_ast.body[0]
    if not isinstance(f_def, ast.FunctionDef):
        raise Exception(f"Movement clause: not a valid function")
    f_name = f_def.name

    prd = Predicates.daylist(days=clause_spec.days,
                             step=clause_spec.leave_step)
    ret = TickDelta(days=clause_spec.duration.to_days(),
                    step=clause_spec.return_step)

    num_args = len(f_def.args.args)
    if num_args == 2:
        f_shape = GeneralClause.by_row
    elif num_args == 3:
        f_shape = GeneralClause.by_cross
    else:
        raise Exception(
            f"Movement clause: invalid number of arguments ({num_args})")

    def compile_clause(ctx: SimContext, global_namespace: dict) -> Clause:
        code = compile(f_ast, '<string>', mode='exec')
        local_namespace: dict[str, Any] = {}
        exec(code, global_namespace, local_namespace)
        f = local_namespace[f_name]
        return f_shape(ctx, f_name, prd, ret, f)

    return compile_clause


def load_movement_spec(spec_string: str) -> MovementBuilder:
    results = movement_spec.parse_string(spec_string, parse_all=True)
    spec: MovementSpec = results[0]  # type: ignore

    clause_compilers = map(parse_clause, spec.clauses)

    def compile_clause(ctx: SimContext) -> Clause:
        global_namespace: dict[str, Any] = {
            'geo': ctx.geo,
            'param': ctx.param,
            'poisson': ctx.rng.poisson,
            'binomial': ctx.rng.binomial,
            'multinomial': ctx.rng.multinomial
        }
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
