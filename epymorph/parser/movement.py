from __future__ import annotations

from typing import NamedTuple

import pyparsing as P

from epymorph.parser.common import code_comment
from epymorph.parser.move_clause import Daily, daily
from epymorph.parser.move_predef import Predef, predef
from epymorph.parser.move_steps import MoveSteps, move_steps


class MovementSpec(NamedTuple):
    steps: MoveSteps
    predef: Predef | None
    clauses: list[Daily]


movement_spec = P.OneOrMore(
    move_steps |
    predef |
    daily
).ignore(code_comment)


@movement_spec.set_parse_action
def marshal_movement(instring: str, loc: int, results: P.ParseResults):
    s = [x for x in results if isinstance(x, MoveSteps)]
    c = [x for x in results if isinstance(x, Daily)]
    p = [x for x in results if isinstance(x, Predef)]
    if len(s) < 1 or len(s) > 1:
        raise P.ParseException(instring, loc,
                               f"Invalid movement specification: expected 1 steps clause, but found {len(s)}.")
    elif len(c) < 1:
        raise P.ParseException(instring, loc,
                               f"Invalid movement specification: expected more than one movement clause, but found 0.")
    elif len(p) > 1:
        raise P.ParseException(
            instring, loc, f"Invalid movement specification: expected 0 or 1 predef clause, but found {len(p)}")

    predef = p[0] if len(p) == 1 else None
    return P.ParseResults(MovementSpec(steps=s[0], predef=predef, clauses=c))
