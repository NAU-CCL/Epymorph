from __future__ import annotations

from functools import reduce
from typing import NamedTuple

import pyparsing as P

E = P.ParserElement
_ = P.Suppress
l = P.Literal
fnumber = P.pyparsing_common.fnumber
fraction = P.pyparsing_common.fraction
integer = P.pyparsing_common.integer


code_comment: E = P.AtLineStart('#') + ... + P.LineEnd()


def field(field_name: str, value_parser: E) -> E:
    return _(l(field_name) + l('=')) + value_parser


def bracketed(value_parser: E) -> E:
    return _('[') + value_parser + _(']')


def tag(tag_name: str, fields: list[E]) -> E:
    def combine(p1, p2):
        return p1 + _(';') + p2
    field_list = reduce(combine, fields[1:], fields[0])
    return bracketed(_(l(tag_name) + l(":")) - field_list)


num_list: E = bracketed(P.delimited_list(fraction | fnumber))


class Duration(NamedTuple):
    value: int
    unit: str

    def to_days(self) -> int:
        if self.unit == 'd':
            return self.value
        elif self.unit == 'w':
            return self.value * 7
        elif self.unit == 'm':
            return self.value * 30  # TODO: this probably isn't correct handling for this case
        else:
            raise Exception(f"unsupported unit {self.unit}")


duration = P.Group(integer + P.one_of('d w m'))


@duration.set_parse_action
def marshal_duration(results: P.ParseResults):
    [value, unit] = results[0]
    return P.ParseResults(Duration(value, unit))


fn_body = P.SkipTo(P.AtLineStart(']'))\
    .set_parse_action(lambda toks: toks.as_list()[0].strip())
