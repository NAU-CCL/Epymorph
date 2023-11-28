from typing import NamedTuple

import pyparsing as P

from epymorph.parser.common import field, integer, num_list, tag


class MoveSteps(NamedTuple):
    """The data model for a MoveSteps clause."""

    step_lengths: list[float]
    """The lengths of each tau step. This should sum to 1."""


move_steps: P.ParserElement = tag('move-steps', [
    field('per-day', integer('num_steps')),
    field('duration', num_list('steps'))
])


@move_steps.set_parse_action
def marshal_move_steps(results: P.ParseResults):
    fields = results.as_dict()
    return MoveSteps(fields['steps'])
