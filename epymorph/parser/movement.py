from __future__ import annotations

from typing import NamedTuple

import pyparsing as P

from epymorph.parser.common import code_comment
from epymorph.parser.move_clause import Daily, daily
from epymorph.parser.move_steps import MoveSteps, move_steps


class MovementSpec(NamedTuple):
    steps: MoveSteps
    clauses: list[Daily]


movement_spec = P.OneOrMore(
    move_steps |
    daily
).ignore(code_comment)


@movement_spec.set_parse_action
def marshal_movement(instring: str, loc: int, results: P.ParseResults):
    s = [x for x in results if isinstance(x, MoveSteps)]
    c = [x for x in results if isinstance(x, Daily)]
    if len(s) < 1 or len(s) > 1:
        raise P.ParseException(instring, loc,
                               f"Invalid movement specification: expected 1 steps clause, but found {len(s)}.")
    elif len(c) < 1:
        raise P.ParseException(instring, loc,
                               f"Invalid movement specification: expected more than one movement clause, but found 0.")
    return P.ParseResults(MovementSpec(steps=s[0], clauses=c))
