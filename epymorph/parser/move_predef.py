from __future__ import annotations

from typing import NamedTuple

import pyparsing as P

from epymorph.parser.common import field, fn_body, tag


class Predef(NamedTuple):
    f: str


predef: P.ParserElement = tag('predef', [field('function', fn_body('f'))])


@predef.set_parse_action
def marshal_predef(results: P.ParseResults):
    fields = results.as_dict()
    return Predef(fields['f'])
