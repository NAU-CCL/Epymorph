from __future__ import annotations

from typing import NamedTuple

import pyparsing as P

from epymorph.parser.common import field, fn_body, tag


class Predef(NamedTuple):
    """The data model of a predef clause."""
    function: str


predef: P.ParserElement = tag('predef', [field('function', fn_body('function'))])
"""The parser for a predef clause."""


@predef.set_parse_action
def marshal_predef(results: P.ParseResults):
    """Convert a pyparsing result into a Predef."""
    fields = results.as_dict()
    return Predef(fields['function'])
