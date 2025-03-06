"""
Methods for querying SODA APIs.
See: https://dev.socrata.com/

There is a package called sodapy that was created for this purpose;
however at time of writing, it is no longer maintained.
"""

from abc import ABC
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date
from itertools import chain
from typing import Generator, Iterable, Literal, Sequence
from urllib.parse import quote, urlencode

import numpy as np
import pandas as pd
from numpy.typing import DTypeLike


def _parens(s: str) -> str:
    return f"({s})"


def _col(s: str) -> str:
    return f"`{s}`"


def _txt(s: str) -> str:
    return f"'{s}'"


def _date(d: date) -> str:
    return f"'{d}T00:00:00'"


@dataclass(frozen=True)
class SocrataResource:
    domain: str
    id: str
    format: Literal["csv"] = field(default="csv")

    @property
    def url(self) -> str:
        return f"https://{self.domain}/resource/{self.id}.{self.format}"


@dataclass(frozen=True, slots=True)
class Select:
    name: str
    """Column name."""
    dtype: Literal["str", "int", "float", "date"]
    """The data type of the column."""
    as_name: str | None = field(default=None)

    def __str__(self) -> str:
        if self.as_name is not None:
            return f"{_col(self.name)} AS {_col(self.as_name)}"
        return _col(self.name)

    @property
    def result_name(self) -> str:
        return self.as_name or self.name

    @property
    def dtype_as_np(self) -> DTypeLike:
        match self.dtype:
            case "str":
                return np.str_
            case "int":
                return np.int64
            case "float":
                return np.float64
            case "date":
                return np.datetime64


class WhereClause(ABC):
    pass


@dataclass(frozen=True, slots=True)
class Equals(WhereClause):
    column: str
    """Column name."""
    value: str
    """The value to test."""

    def __str__(self) -> str:
        return f"{_col(self.column)}={_txt(self.value)}"


@dataclass(frozen=True, slots=True)
class In(WhereClause):
    column: str
    """Column name."""
    values: Iterable[str]
    """The sequence of values to test."""

    def __post_init__(self):
        if isinstance(self.values, Iterator):
            values = list(self.values)
            object.__setattr__(self, "values", values)

    def __str__(self) -> str:
        values_list = ",".join([_txt(x) for x in self.values])
        return f"{_col(self.column)} IN ({values_list})"


@dataclass(frozen=True, slots=True)
class DateBetween(WhereClause):
    column: str
    """Column name."""
    start: date
    """Start date."""
    end: date
    """End date (inclusive)."""

    def __str__(self) -> str:
        return f"{_col(self.column)} BETWEEN {_date(self.start)} AND {_date(self.end)}"


@dataclass(frozen=True, slots=True)
class And(WhereClause):
    clauses: Sequence[WhereClause]
    """The clauses to join with an 'AND'."""

    def __init__(self, *args: WhereClause | Sequence[WhereClause]):
        clauses = list(
            chain.from_iterable(
                ([x] if isinstance(x, WhereClause) else x for x in args)
            )
        )
        object.__setattr__(self, "clauses", clauses)

    def __str__(self) -> str:
        match len(self.clauses):
            case 0:
                return ""
            case 1:
                return str(self.clauses[0])
            case _:
                return _parens(" AND ".join([str(x) for x in self.clauses]))


@dataclass(frozen=True, slots=True)
class Or(WhereClause):
    clauses: Sequence[WhereClause]
    """The clauses to join with an 'OR'."""

    def __init__(self, *args: WhereClause | Sequence[WhereClause]):
        clauses = list(
            chain.from_iterable(
                ([x] if isinstance(x, WhereClause) else x for x in args)
            )
        )
        object.__setattr__(self, "clauses", clauses)

    def __str__(self) -> str:
        match len(self.clauses):
            case 0:
                return ""
            case 1:
                return str(self.clauses[0])
            case _:
                return _parens(" OR ".join([str(x) for x in self.clauses]))


@dataclass(frozen=True, slots=True)
class Not(WhereClause):
    clause: WhereClause
    """The clause to invert."""

    def __str__(self) -> str:
        return f"NOT {str(self.clause)}"


class OrderClause(ABC):
    pass


@dataclass(frozen=True, slots=True)
class Ascending(OrderClause):
    column: str
    """The column name."""

    def __str__(self) -> str:
        return f"{_col(self.column)} ASC"


@dataclass(frozen=True, slots=True)
class Descending(OrderClause):
    column: str
    """The column name."""

    def __str__(self) -> str:
        return f"{_col(self.column)} DESC"


def query_csv(
    resource: SocrataResource,
    select: Sequence[Select],
    where: WhereClause,
    *,
    order_by: Sequence[OrderClause] | None = None,
    limit: int = 10000,
    api_token: str | None = None,
) -> pd.DataFrame:
    query = {
        "$select": ",".join([str(x) for x in select]),
        "$where": str(where),
        "$limit": limit,
    }
    if order_by is not None and len(order_by) > 0:
        query["$order"] = ",".join([str(x) for x in order_by])

    soql = urlencode(quote_via=quote, safe=",()'$:", query=query)
    query_url = f"{resource.url}?{soql}"

    # TODO: this handling of types is rudimentary --
    # type conversion is essentially non-existent, e.g., '3.0' will raise an error
    # if specified to be an int column.
    # Okay for now, but definitely something that could be improved
    # and we'll want to at least catch errors helpfully.

    column_dtypes = {x.result_name: x.dtype_as_np for x in select if x.dtype != "date"}
    column_parsedates = [x.result_name for x in select if x.dtype == "date"]
    req_headers = None if api_token is None else {"X-App-Token": api_token}

    def query_frames(offset: int = 0) -> Generator[pd.DataFrame, None, None]:
        try:
            frame_df = pd.read_csv(
                f"{query_url}&$offset={offset}",
                dtype=column_dtypes,  # type: ignore
                parse_dates=column_parsedates,
                storage_options=req_headers,
            )
        except ValueError as e:
            # TODO
            raise e
        yield frame_df

        if (frame_size := len(frame_df.index)) >= limit:
            yield from query_frames(offset + frame_size)

    return pd.concat(query_frames())
