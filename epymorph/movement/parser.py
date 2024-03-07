"""Parsing of MovementSpecs."""
from typing import Literal, NamedTuple, cast

import pyparsing as P
from numpy.typing import DTypeLike

import epymorph.movement.parser_util as p
from epymorph.data_shape import DataShape
from epymorph.error import MmParseException

# A MovementSpec has the following object structure:
#
# - MovementSpec
#   - MoveSteps (1)
#   - Attribute (0 or more)
#   - Predef (0 or 1)
#   - MovementClause (1 or more)

############################################################
# MoveSteps
############################################################


class MoveSteps(NamedTuple):
    """The data model for a MoveSteps clause."""
    step_lengths: list[float]
    """The lengths of each tau step. This should sum to 1."""


move_steps: P.ParserElement = p.tag('move-steps', [
    p.field('per-day', p.integer)('num_steps'),
    p.field('duration', p.num_list)('steps')
])


@move_steps.set_parse_action
def marshal_move_steps(results: P.ParseResults):
    """Convert a pyparsing result to a MoveSteps."""
    fields = results.as_dict()
    return MoveSteps(fields['steps'])


############################################################
# Attributes
############################################################


class Attribute(NamedTuple):
    """The data model for an Attribute clause."""
    # TODO: maybe replace with epymorph.simulation.AttributeDef in v0.5
    name: str
    shape: DataShape
    dtype: DTypeLike  # TODO: replace with DataDType in v0.5?
    source: Literal['geo', 'params']
    description: str


attribute: P.ParserElement = p.tag('attrib', [
    p.field('source', P.one_of('geo params')),
    p.field('name', p.name),
    p.field('shape', p.shape),
    p.field('dtype', p.dtype),
    p.field('description', p.quoted),
])


@attribute.set_parse_action
def marshal_attribute(results: P.ParseResults):
    """Convert a pyparsing result to an Attribute."""
    fields = results.as_dict()
    return Attribute(fields['name'], fields['shape'], fields['dtype'][0], fields['source'], fields['description'])


############################################################
# Predef
############################################################


class Predef(NamedTuple):
    """The data model of a predef clause."""
    function: str


predef: P.ParserElement = p.tag('predef', [p.field('function', p.fn_body('function'))])
"""The parser for a predef clause."""


@predef.set_parse_action
def marshal_predef(results: P.ParseResults):
    """Convert a pyparsing result into a Predef."""
    fields = results.as_dict()
    return Predef(fields['function'])


############################################################
# MovementClause
############################################################


day_list: P.ParserElement = p.bracketed(
    P.delimited_list(P.one_of('M T W Th F Sa Su'))
)
"""Parser for a square-bracketed list of days-of-the-week."""


DayOfWeek = Literal['M', 'T', 'W', 'Th', 'F', 'Sa', 'Su']
"""Type for days of the week values."""

ALL_DAYS: list[DayOfWeek] = ['M', 'T', 'W', 'Th', 'F', 'Sa', 'Su']
"""A list of all days of the week values."""


class DailyClause(NamedTuple):
    """
    The data model for a daily movement clause.
    Note: leave_step and return_step should be 0-indexed in this form.
    """
    # Conversion from 1-indexed steps happens during parsing.
    days: list[DayOfWeek]
    leave_step: int
    duration: p.Duration
    return_step: int
    function: str


daily: P.ParserElement = p.tag('mtype', [
    p.field('days', ('all' | day_list)),
    p.field('leave', p.integer),
    p.field('duration', p.duration),
    p.field('return', p.integer),
    p.field('function', p.fn_body)
])
"""
Parser for a DailyClause. e.g.:
```
[mtype: days=[all]; leave=0; duration=7d; return=0; function=
def simple_movement(t, src, dst):
    return 10
]
```
"""


@daily.set_parse_action
def marshal_daily(instring: str, loc: int, results: P.ParseResults):
    """Convert a pyparsing result to a Daily."""
    fields = results.as_dict()
    days = fields['days']
    leave_step = fields['leave']
    duration = fields['duration'][0]
    return_step = fields['return']
    function = fields['function']
    if not isinstance(days, list):
        msg = f"Unsupported value for movement clause daily: days ({days})"
        raise P.ParseException(instring, loc, msg)
    elif days == ['all']:
        days = ALL_DAYS.copy()
    else:
        days = cast(list[DayOfWeek], days)
    if leave_step < 1:
        msg = f"movement clause daily: leave step must be at least 1 (value was: {leave_step})"
        raise P.ParseException(instring, loc, msg)
    if return_step < 1:
        msg = f"movement clause daily: return step must be at least 1 (value was: {leave_step})"
        raise P.ParseException(instring, loc, msg)
    return DailyClause(days, leave_step - 1, duration, return_step - 1, function)


MovementClause = DailyClause
"""Data classes representing all possible movement clauses."""


############################################################
# MovementSpec
############################################################


class MovementSpec(NamedTuple):
    """The data model for a movement model spec."""
    steps: MoveSteps
    attributes: list[Attribute]
    predef: Predef | None
    clauses: list[MovementClause]


def parse_movement_spec(string: str) -> MovementSpec:
    """Parse a MovementSpec from the given string."""
    try:
        result = movement_spec.parse_string(string, parse_all=True)
        return cast(MovementSpec, result[0])
    except Exception as e:
        msg = "Unable to parse MovementModel."
        raise MmParseException(msg) from e


movement_spec = P.OneOrMore(
    move_steps |
    attribute |
    predef |
    daily
).ignore(p.code_comment)
"""The parser for MovementSpec."""


@movement_spec.set_parse_action
def marshal_movement(instring: str, loc: int, results: P.ParseResults):
    """Convert a pyparsing result to a MovementSpec."""
    s = [x for x in results if isinstance(x, MoveSteps)]
    a = [x for x in results if isinstance(x, Attribute)]
    c = [x for x in results if isinstance(x, MovementClause)]
    d = [x for x in results if isinstance(x, Predef)]
    if len(s) < 1 or len(s) > 1:
        msg = f"Invalid movement specification: expected 1 steps clause, but found {len(s)}."
        raise P.ParseException(instring, loc, msg)
    if len(c) < 1:
        msg = "Invalid movement specification: expected more than one movement clause, but found 0."
        raise P.ParseException(instring, loc, msg)
    if len(d) > 1:
        msg = f"Invalid movement specification: expected 0 or 1 predef clause, but found {len(d)}"
        raise P.ParseException(instring, loc, msg)

    predef_clause = d[0] if len(d) == 1 else None
    return P.ParseResults(MovementSpec(steps=s[0], attributes=a, predef=predef_clause, clauses=c))
