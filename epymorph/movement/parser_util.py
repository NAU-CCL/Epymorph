"""
Common parsing utilities.
"""
from functools import reduce
from typing import NamedTuple

import numpy as np
import pyparsing as P

from epymorph.data_shape import parse_shape

# It's likely this will need to move to a different package (i.e., not `movement`)
# but it's fine here for now since movement is the only thing with a spec parser.

E = P.ParserElement
_ = P.Suppress
l = P.Literal
fnumber = P.pyparsing_common.fnumber
fraction = P.pyparsing_common.fraction
integer = P.pyparsing_common.integer

quoted = P.QuotedString(quote_char='"', unquote_results=True) |\
    P.QuotedString(quote_char="'", unquote_results=True)
"""Allow both single- or double-quote-delimited strings."""

name: E = P.Word(
    init_chars=P.srange("[a-zA-Z]"),
    body_chars=P.srange("[a-zA-Z0-9_]"),
)
"""A name string, suitable for use as a Python variable name, for instance."""

code_comment: E = P.AtLineStart('#') + ... + P.LineEnd()
"""Parser for Python-style code comments."""


def field(field_name: str, value_parser: E) -> E:
    """Parser for a clause field, like `<field_name>=<value>`"""
    return _(l(field_name) + l('=')) + value_parser.set_results_name(field_name)


def bracketed(value_parser: E) -> E:
    """Wrap another parser to surround it with square brackets."""
    return _('[') + value_parser + _(']')


def tag(tag_name: str, fields: list[E]) -> E:
    """
    Parser for a spec tag: this is a top-level item, surrounded by square brackets,
    identified by a tag name and containing one-or-more fields. e.g.:
    `[<tag_name>: <field_1>=<value>; <field_2>=<value>]`
    """
    def combine(p1, p2):
        return p1 + _(';') + p2
    field_list = reduce(combine, fields[1:], fields[0])
    return bracketed(_(l(tag_name) + l(":")) - field_list)


num_list: E = bracketed(P.delimited_list(fraction | fnumber))
"""Parser for a list of numbers in brackets, like: `[1,2,3]`"""


class Duration(NamedTuple):
    """Data class for a duration expression."""
    value: int
    unit: str

    def to_days(self) -> int:
        """Return the equivalent number of days."""
        if self.unit == 'd':
            return self.value
        elif self.unit == 'w':
            return self.value * 7
        else:
            raise ValueError(f"unsupported unit {self.unit}")


duration = P.Group(integer + P.one_of('d w'))
"""Parser for a duration expression."""


@duration.set_parse_action
def marshal_duration(results: P.ParseResults):
    """Convert a pyparsing result to a Duration object."""
    [value, unit] = results[0]
    return P.ParseResults(Duration(value, unit))


fn_body = P.SkipTo(P.AtLineStart(']'))\
    .set_parse_action(lambda toks: toks.as_list()[0].strip())
"""
Parser for a function body, which runs until the end of a clause ends.
For example, if a tag value should be a function, you can define it like this:
`p.field('function', p.fn_body('function'))` which can be used to parse things like:
```
[<tag-name>: function=
def my_function():
    return 'hello world'
]
```
"""


shape = P.one_of("S T N TxN NxN")
"""
Describes the dimensions of an array in terms of the simulation.
For example "TxN" describes a two-dimensional array which is the
number of simulation days in the first dimensions and the number
of geo nodes in the second dimension. See `epymorph.data_shape` for more info.
(Excludes Shapes with arbitrary dimensions for simplicity.)
"""


@shape.set_parse_action
def marshal_shape(results: P.ParseResults):
    """Convert a pyparsing result to a Shape object."""
    value = str(results[0])
    return P.ParseResults(parse_shape(value))


base_dtype = P.one_of('int float str')
"""One of epymorph's base permitted data types."""


@base_dtype.set_parse_action
def marshal_base_dtype(results: P.ParseResults):
    """Convert a pyparsing result to dtype object."""
    match results[0]:
        case 'int':
            result = np.int64
        case 'float':
            result = np.float64
        case 'str':
            result = np.str_
    return result


struct_dtype_field = _('(') + name('name') + _(',') + base_dtype('dtype') + _(')')
"""A single named field in a structured dtype."""


@struct_dtype_field.set_parse_action
def marshal_struct_dtype_field(results: P.ParseResults):
    """Convert a pyparsing result to a tuple representing a single named field in a structured dtype."""
    return (results['name'], results['dtype'])


struct_dtype = bracketed(P.delimited_list(struct_dtype_field))
"""A complete structured dtype."""


@struct_dtype.set_parse_action
def marshal_struct_dtype(results: P.ParseResults):
    """Convert a pyparsing results to a structured dtype object."""
    return np.dtype(results.as_list())


dtype = base_dtype | struct_dtype
