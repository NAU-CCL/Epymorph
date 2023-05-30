from __future__ import annotations

from typing import Literal, NamedTuple, cast

import pyparsing as P

from epymorph.parser.common import (Duration, bracketed, duration, field,
                                    fn_body, integer, tag)

day_list: P.ParserElement = bracketed(
    P.delimited_list(P.one_of('M T W Th F Sa Su'))
)


DayOfWeek = Literal['M', 'T', 'W', 'Th', 'F', 'Sa', 'Su']

ALL_DAYS: list[DayOfWeek] = ['M', 'T', 'W', 'Th', 'F', 'Sa', 'Su']


class Daily(NamedTuple):
    days: list[DayOfWeek]
    leave_step: int
    duration: Duration
    return_step: int
    f: str


daily: P.ParserElement = tag('mtype', [
    field('days', ('all' | day_list)('days')),
    field('leave', integer('leave')),
    field('duration', duration('duration')),
    field('return', integer('return')),
    field('function', fn_body('f'))
])


@daily.set_parse_action
def marshal_daily(results: P.ParseResults):
    fields = results.as_dict()
    days = fields['days']
    if not isinstance(days, list):
        raise Exception(
            f"Unsupported value for movement clause daily: days ({days})")
    elif days == ['all']:
        days = ALL_DAYS.copy()
    else:
        days = cast(list[DayOfWeek], days)
    return Daily(
        days,
        fields['leave'],
        fields['duration'][0],
        fields['return'],
        fields['f'])
