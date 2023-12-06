"""
Common parsing utilities.
"""
from functools import reduce
from typing import NamedTuple

import pyparsing as P

# It's likely this will need to move to a different package (i.e., not `movement`)
# but it's fine here for now since movement is the only thing with a spec parser.

E = P.ParserElement
_ = P.Suppress
l = P.Literal
fnumber = P.pyparsing_common.fnumber
fraction = P.pyparsing_common.fraction
integer = P.pyparsing_common.integer


code_comment: E = P.AtLineStart('#') + ... + P.LineEnd()
"""Parser for Python-style code comments."""


def field(field_name: str, value_parser: E) -> E:
    """Parser for a clause field, like `<field_name>=<value>`"""
    return _(l(field_name) + l('=')) + value_parser


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
