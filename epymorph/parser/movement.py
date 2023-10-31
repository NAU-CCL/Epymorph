from __future__ import annotations

from typing import NamedTuple, cast

import pyparsing as P

from epymorph.error import MmValidationException
from epymorph.parser.common import code_comment
from epymorph.parser.move_clause import MovementClause, daily
from epymorph.parser.move_predef import Predef, predef
from epymorph.parser.move_steps import MoveSteps, move_steps


class MovementSpec(NamedTuple):
    """The data model for a movement model spec."""
    steps: MoveSteps
    predef: Predef | None
    clauses: list[MovementClause]

    @staticmethod
    def load(string: str) -> MovementSpec:
        """Parse a MovementSpec from the given string."""
        try:
            result = movement_spec.parse_string(string, parse_all=True)
            return cast(MovementSpec, result[0])
        except Exception as e:
            msg = "Unable to parse MovementModel."
            raise MmValidationException(msg) from e


movement_spec = P.OneOrMore(
    move_steps |
    predef |
    daily
).ignore(code_comment)
"""The parser for MovementSpec."""


@movement_spec.set_parse_action
def marshal_movement(instring: str, loc: int, results: P.ParseResults):
    """Convert a pyparsing result to a MovementSpec."""
    s = [x for x in results if isinstance(x, MoveSteps)]
    c = [x for x in results if isinstance(x, MovementClause)]
    p = [x for x in results if isinstance(x, Predef)]
    if len(s) < 1 or len(s) > 1:
        raise P.ParseException(instring, loc,
                               f"Invalid movement specification: expected 1 steps clause, but found {len(s)}.")
    if len(c) < 1:
        raise P.ParseException(instring, loc,
                               "Invalid movement specification: expected more than one movement clause, but found 0.")
    if len(p) > 1:
        raise P.ParseException(
            instring, loc, f"Invalid movement specification: expected 0 or 1 predef clause, but found {len(p)}")

    predef_clause = p[0] if len(p) == 1 else None
    return P.ParseResults(MovementSpec(steps=s[0], predef=predef_clause, clauses=c))
