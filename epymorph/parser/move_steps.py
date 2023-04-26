from __future__ import annotations

from typing import NamedTuple

import pyparsing as P

from epymorph.parser.common import field, integer, num_list, tag


class MoveSteps(NamedTuple):
    steps: list[float]


move_steps: P.ParserElement = tag('move-steps', [
    field('per-day', integer('num_steps')),
    field('duration', num_list('steps'))
])


@move_steps.set_parse_action
def marshal_move_steps(results: P.ParseResults):
    fields = results.as_dict()
    return MoveSteps(fields['steps'])
