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
from typing import Callable, Generator, Iterable, Literal, Sequence, TypeVar
from urllib.parse import quote, urlencode

import numpy as np
import pandas as pd
from numpy.typing import DTypeLike

T_co = TypeVar("T_co", covariant=True)


def _parens(s: str) -> str:
    return f"({s})"


def _col(s: str) -> str:
    return f"`{s}`"


def _txt(s: str) -> str:
    return f"'{s}'"


def _date(d: date) -> str:
    return f"'{d}T00:00:00'"


def _list(xs: Iterable[T_co], to_string: Callable[[T_co], str] = str) -> str:
    return ",".join([to_string(x) for x in xs])


ColumnType = Literal["str", "int", "nullable_int", "float", "date", "bool"]


def to_pandas_type(col_type: ColumnType) -> DTypeLike:
    match col_type:
        case "str":
            return np.str_
        case "int":
            return np.int64
        case "nullable_int":
            # allow integer columns to use Pandas' nullable integer type
            return "Int64"
        case "float":
            return np.float64
        case "date":
            return np.datetime64
        case "bool":
            return np.bool_


@dataclass(frozen=True)
class SocrataResource:
    domain: str
    id: str

    @property
    def url(self) -> str:
        return f"https://{self.domain}/resource/{self.id}"


@dataclass(frozen=True, slots=True)
class Select:
    name: str
    """Column name."""
    dtype: ColumnType
    """The data type of the column."""
    as_name: str | None = field(default=None)
    """Define a new name for the column; the 'AS' statement."""

    def __str__(self) -> str:
        if self.as_name is not None:
            return f"{_col(self.name)} AS {_col(self.as_name)}"
        return _col(self.name)

    @property
    def result_name(self) -> str:
        return self.as_name or self.name


@dataclass(frozen=True, slots=True)
class SelectExpression:
    expression: str
    """Expression; as best practice you should escape column names in the expression
    by surrounding them in back-ticks."""
    dtype: ColumnType
    """The data type of the column."""
    as_name: str
    """Define a name for the result; the 'AS' statement."""

    def __str__(self) -> str:
        return f"({self.expression}) AS {_col(self.as_name)}"

    @property
    def result_name(self) -> str:
        return self.as_name


SelectStatements = Select | SelectExpression


class WhereClause(ABC):
    pass


@dataclass(frozen=True, slots=True)
class NotNull(WhereClause):
    column: str
    """Column name."""

    def __str__(self) -> str:
        return f"{_col(self.column)} IS NOT NULL"


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
        return f"{_col(self.column)} IN ({_list(self.values, _txt)})"


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

    def __init__(self, *clauses: WhereClause):
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

    def __init__(self, *clauses: WhereClause):
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


@dataclass(frozen=True, slots=True)
class Query:
    select: Sequence[SelectStatements]
    where: WhereClause
    order_by: Sequence[OrderClause] | None = field(default=None)

    def __str__(self) -> str:
        soql = f"SELECT {_list(self.select)} WHERE {self.where}"
        if self.order_by is not None and len(self.order_by) > 0:
            return f"{soql} ORDER BY {_list(self.order_by)}"
        return soql


def query_csv(
    resource: SocrataResource,
    query: Query,
    *,
    limit: int = 10000,
    result_filter: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    api_token: str | None = None,
) -> pd.DataFrame:
    return query_csv_soql(
        resource=resource,
        soql=str(query),
        column_types=[(x.result_name, x.dtype) for x in query.select],
        limit=limit,
        result_filter=result_filter,
        api_token=api_token,
    )


def query_csv_soql(
    resource: SocrataResource,
    soql: str,
    column_types: Sequence[tuple[str, ColumnType]],
    *,
    limit: int = 10000,
    result_filter: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    api_token: str | None = None,
) -> pd.DataFrame:
    # TODO: this handling of types is rudimentary --
    # type conversion is essentially non-existent, e.g., '3.0' will raise an error
    # if specified to be an int column.
    # Okay for now, but definitely something that could be improved
    # and we'll want to at least catch errors helpfully.

    column_dtypes = {n: to_pandas_type(t) for n, t in column_types if t != "date"}
    column_parsedates = [n for n, t in column_types if t == "date"]
    req_headers = None if api_token is None else {"X-App-Token": api_token}

    def query_pages(offset: int = 0) -> Generator[pd.DataFrame, None, None]:
        page_soql = f"{soql} LIMIT {limit} OFFSET {offset}"
        query_string = urlencode(
            quote_via=quote,
            safe=",()'$:",
            query={"$query": page_soql},
        )
        page_df = pd.read_csv(
            f"{resource.url}.csv?{query_string}",
            dtype=column_dtypes,  # type: ignore
            parse_dates=column_parsedates,
            storage_options=req_headers,
            date_format="ISO8601",
        )
        response_rows = len(page_df.index)
        if result_filter is not None:
            page_df = result_filter(page_df)
        yield page_df

        if (page_size := response_rows) >= limit:
            yield from query_pages(offset + page_size)

    return pd.concat(query_pages())
