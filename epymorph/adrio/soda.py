"""
Methods for querying SODA APIs.

See Also
--------
The API documentation at [https://dev.socrata.com/](https://dev.socrata.com/).

Examples
--------
--8<-- "docs/_examples/adrio_soda.md"
"""

# NOTE: There is a package called sodapy that was created for this purpose;
# however at time of writing, it is no longer maintained.

from abc import abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date
from typing import (
    Any,
    Callable,
    Generator,
    Iterable,
    Literal,
    Protocol,
    Sequence,
    TypeVar,
)
from urllib.parse import quote, urlencode

import numpy as np
import pandas as pd
import requests
from numpy.typing import DTypeLike
from typing_extensions import override


def _parens(s: str) -> str:
    """Formatting util: enclose in parens."""  # noqa: D401
    return f"({s})"


def _col(s: str) -> str:
    """Formatting util: SOQL column name."""  # noqa: D401
    return f"`{s}`"


def _txt(s: str) -> str:
    """Formatting util: SOQL text."""  # noqa: D401
    return f"'{s}'"


def _date(d: date) -> str:
    """Formatting util: date as SOQL timestamp."""  # noqa: D401
    return f"'{d}T00:00:00'"


_T_co = TypeVar("_T_co", covariant=True)


def _list(xs: Iterable[_T_co], to_string: Callable[[_T_co], str] = str) -> str:
    """Formatting util: list of values."""  # noqa: D401
    return ",".join([to_string(x) for x in xs])


ColumnType = Literal["str", "int", "nullable_int", "float", "date", "bool"]
"""Simplified set of types for SOQL result columns."""


def _to_pandas_type(col_type: ColumnType) -> DTypeLike:
    """Get the pandas type that corresponds to the SOQL column type."""
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
    """
    Defines a Socrata API resource.

    Parameters
    ----------
    domain :
        The domain where the API is hosted.
    id :
        The ID of the resource.
    """

    domain: str
    """The domain where the API is hosted."""
    id: str
    """The ID of the resource."""

    @property
    def url(self) -> str:
        """The URL for this resource."""
        return f"https://{self.domain}/resource/{self.id}"

    @property
    def metadata_url(self) -> str:
        """The URL for the metadata description of the resource (JSON format)."""
        return f"https://{self.domain}/api/views/{self.id}.json"


# ARCHITECTURE NOTE: what follows is a collection of objects which model a SOQL query.
# Compose your query by putting together instances of these clause types.
# The whole query can be converted to a string -- `str(query_object)` -- which
# in turn calls `str` on all of the clauses as appropriate and combines the
# parts into a full query.


class SelectClause(Protocol):
    """The common interface for SOQL select clauses."""

    @property
    @abstractmethod
    def result_name(self) -> str:
        """The name to use to refer to the result."""

    @property
    @abstractmethod
    def result_dtype(self) -> ColumnType:
        """The data type of the result."""
        # This is really just here to avoid issues with dataclasses and inheritence.


@dataclass(frozen=True, slots=True)
class Select(SelectClause):
    """
    A SOQL select clause for selecting a column as-is.

    Parameters
    ----------
    name :
        Column name.
    dtype :
        The data type of the column.
    as_name :
        Define a new name for the column; the 'AS' statement.
    """

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
    @override
    def result_name(self) -> str:
        return self.as_name or self.name

    @property
    @override
    def result_dtype(self) -> ColumnType:
        return self.dtype


@dataclass(frozen=True, slots=True)
class SelectExpression(SelectClause):
    """
    A SOQL select clause for selecting the result of an expression.

    Parameters
    ----------
    expression :
        Expression; as best practice you should escape column names in the expression
        by surrounding them in back-ticks.
    dtype :
        The data type of the column.
    as_name :
        Define a name for the result; the 'AS' statement.
    """

    expression: str
    """
    Expression; as best practice you should escape column names in the expression
    by surrounding them in back-ticks.
    """
    dtype: ColumnType
    """The data type of the column."""
    as_name: str
    """Define a name for the result; the 'AS' statement."""

    def __str__(self) -> str:
        return f"({self.expression}) AS {_col(self.as_name)}"

    @property
    @override
    def result_name(self) -> str:
        return self.as_name

    @property
    @override
    def result_dtype(self) -> ColumnType:
        return self.dtype


class WhereClause(Protocol):
    """The common interface for SOQL where clauses."""


@dataclass(frozen=True, slots=True)
class NotNull(WhereClause):
    """
    A where-clause for rows that are not null.

    Parameters
    ----------
    column :
        Column name.
    """

    column: str
    """Column name."""

    def __str__(self) -> str:
        return f"{_col(self.column)} IS NOT NULL"


@dataclass(frozen=True, slots=True)
class Equals(WhereClause):
    """
    A where-clause for rows whose values are equal to the given value.

    Parameters
    ----------
    column :
        Column name.
    value :
        The value to test.
    """

    column: str
    """Column name."""
    value: str
    """The value to test."""

    def __str__(self) -> str:
        return f"{_col(self.column)}={_txt(self.value)}"


@dataclass(frozen=True, slots=True)
class In(WhereClause):
    """
    A where-clause for rows whose values are in the given set of values.

    Parameters
    ----------
    column :
        Column name.
    values :
        The sequence of values to test.
    """

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
    """
    A where-clause for rows whose date values are between two dates.
    Endpoints are inclusive.

    Parameters
    ----------
    column :
        Column name.
    start :
        Start date.
    end :
        End date (inclusive).
    """

    column: str
    """Column name."""
    start: date
    """Start date."""
    end: date
    """End date (inclusive)."""

    def __str__(self) -> str:
        return f"{_col(self.column)} BETWEEN {_date(self.start)} AND {_date(self.end)}"


@dataclass(frozen=True, slots=True, init=False)
class And(WhereClause):
    """
    A where-clause that joins other where clauses with "and".

    Parameters
    ----------
    clauses :
        The clauses to join with an 'AND'.
    """

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


@dataclass(frozen=True, slots=True, init=False)
class Or(WhereClause):
    """
    A where-clause that joins other where clauses with "or".

    Parameters
    ----------
    clauses :
        The clauses to join with an 'OR'.
    """

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
    """
    A where-clause that negates another where clause.

    Parameters
    ----------
    clause :
        The clause to negate.
    """

    clause: WhereClause
    """The clause to negate."""

    def __str__(self) -> str:
        return f"NOT {str(self.clause)}"


class OrderClause(Protocol):
    """The common interface for SOQL order-by clauses."""


@dataclass(frozen=True, slots=True)
class Ascending(OrderClause):
    """
    An order-by-clause that sorts the named column in ascending order.

    Parameters
    ----------
    column :
        The column name.
    """

    column: str
    """The column name."""

    def __str__(self) -> str:
        return f"{_col(self.column)} ASC"


@dataclass(frozen=True, slots=True)
class Descending(OrderClause):
    """
    An order-by-clause that sorts the named column in descending order.

    Parameters
    ----------
    column :
        The column name.
    """

    column: str
    """The column name."""

    def __str__(self) -> str:
        return f"{_col(self.column)} DESC"


@dataclass(frozen=True, slots=True)
class Query:
    """
    A complete SOQL query, which includes oen or more select clauses,
    exactly one where clause, and any number of order-by clauses.

    Parameters
    ----------
    select :
        One or more select clauses.
    where :
        A where clause.
    order_by :
        Zero or more order-by clause.
    """

    select: Sequence[SelectClause]
    """One or more select clauses."""
    where: WhereClause
    """A where clause."""
    order_by: Sequence[OrderClause] | None = field(default=None)
    """Zero or more order-by clauses."""

    def __post_init__(self):
        if len(self.select) == 0:
            err = "Query must contain at least one select clause."
            raise ValueError(err)

    def __str__(self) -> str:
        soql = f"SELECT {_list(self.select)} WHERE {self.where}"
        if self.order_by is not None and len(self.order_by) > 0:
            return f"{soql} ORDER BY {_list(self.order_by)}"
        return soql


def get_metadata(resource: SocrataResource) -> Any:
    """
    Retrieve the JSON-formatted metadata for the given resource.

    Parameters
    ----------
    resource :
        The data resource.

    Returns
    -------
    :
        The JSON metadata.
    """
    response = requests.get(resource.metadata_url, timeout=60)
    response.raise_for_status()
    metadata = response.json()
    return metadata


def query_csv(
    resource: SocrataResource,
    query: Query,
    *,
    limit: int = 10000,
    result_filter: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    api_token: str | None = None,
) -> pd.DataFrame:
    """
    Issue `query` against the data in `resource` and return the result as a dataframe.
    For particularly large result sets, we may need to break up the total query into
    more than one request. This is handled automatically, as is concatenating the
    results.

    Parameters
    ----------
    resource :
        The data resource.
    query :
        The query object.
    limit :
        The maximum number of rows fetched per request. Changing this should not change
        the results, but may change the number of requests (and the clock time) it takes
        to produce the results.
    result_filter :
        An optional transform to apply to the results, intended to be used to filter out
        rows from the result in situations where doing this filtering as a where-clause
        would be impossible or impractical. This function will be run on the results of
        each request, not the end result, since it is likely to be more efficient to
        filter as we go when the data requires many requests to complete.
    api_token :
        An optional API token to include in requests.

    Returns
    -------
    :
        The query results.
    """
    return query_csv_soql(
        resource=resource,
        soql=str(query),
        column_types=[(x.result_name, x.result_dtype) for x in query.select],
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
    """
    Issue the query (given in string form, `soql`) against the data in `resource` and
    return the result as a dataframe. For particularly large result sets, we may need to
    break up the total query into more than one request. This is handled automatically,
    as is concatenating the results.

    Parameters
    ----------
    resource :
        The data resource.
    soql :
        The query string.
    column_types :
        The expected data types of the selected columns.
    limit :
        The maximum number of rows fetched per request. Changing this should not change
        the results, but may change the number of requests (and the clock time) it takes
        to produce the results.
    result_filter :
        An optional transform to apply to the results, intended to be used to filter out
        rows from the result in situations where doing this filtering as a where-clause
        would be impossible or impractical. This function will be run on the results of
        each request, not the end result, since it is likely to be more efficient to
        filter as we go when the data requires many requests to complete.
    api_token :
        An optional API token to include in requests.

    Returns
    -------
    :
        The query results.
    """

    # NOTE: type conversion is essentially non-existent,
    # e.g., if the data has '3.0' in a column declared as an int, this will raise.

    column_dtypes = {n: _to_pandas_type(t) for n, t in column_types if t != "date"}
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
