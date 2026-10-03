import itertools
import json
import os
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from functools import cache
from time import perf_counter
from typing import Callable, Literal, NamedTuple, Self, TypeGuard, cast

import numpy as np
import pandas as pd
import requests
from numpy.typing import NDArray
from typing_extensions import override

from epymorph.adrio import acs5
from epymorph.adrio.adrio import (
    ADRIO,
    ADRIOCommunicationError,
    ADRIOContextError,
    ADRIOError,
    ADRIOProcessingError,
    DeferredADRIOError,
    InspectResult,
    PipelineResult,
    adrio_cache,
    adrio_exception_handling,
    adrio_validate_pipe,
)
from epymorph.adrio.validation import (
    ResultFormat,
    validate_dtype,
    validate_numpy,
    validate_shape,
    validate_values_in_range,
)
from epymorph.cache import load_or_fetch_url, module_cache_path
from epymorph.data_shape import Shapes
from epymorph.data_type import AttributeData, StratifiedAttributeArray
from epymorph.error import MissingContextError
from epymorph.geography.us_census import BlockGroupScope, CensusScope
from epymorph.simulation import Context
from epymorph.strata import DEFAULT_STRATA
from epymorph.util import filter_with_mask, split_at


@dataclass
class DataIssues:
    data: pd.DataFrame
    issues: dict[str, NDArray[np.bool_]] = field(default_factory=dict)

    def fix(
        self,
        issue_name: str,
        where: Callable[[pd.DataFrame], pd.DataFrame],
        replacement: int | None = None,
    ) -> Self:
        issue_mask = where(self.data)
        if issue_mask.any(axis=None):
            if replacement is None:
                self.issues[issue_name] = issue_mask.to_numpy()
            else:
                self.data[issue_mask] = replacement
        return self

    def to_result(self) -> PipelineResult:
        return PipelineResult(value=self.data.to_numpy(), issues=self.issues)


######################
# ACS5 API Functions #
######################

# fmt:off
ACS5_YEARS: Sequence[int] = (2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024)  # noqa: E501
"""All supported ACS5 data years."""

RaceCategory = Literal["White", "Black", "Native", "Asian", "Pacific Islander", "Other", "Multiple"]  # noqa: E501
"""A racial category defined by ACS5."""
# fmt: on

_ACS5_CACHE_PATH = module_cache_path(__name__)
"""
For caching ACS5. At the moment, the only thing that is cached is variables metadata.
"""


def census_api_key() -> str | None:
    key = os.environ.get("API_KEY__census.gov", default=None)
    if key is None:
        key = os.environ.get("CENSUS_API_KEY", default=None)
    return key


def url(year: int) -> str:
    return f"https://api.census.gov/data/{year}/acs/acs5"


@cache
def get_vars(year: int) -> dict[str, dict]:
    try:
        vars_url = f"{url(year)}/variables.json"
        cache_path = _ACS5_CACHE_PATH / f"variables-{year}.json"
        file = load_or_fetch_url(vars_url, cache_path)
        return json.load(file)["variables"]
    except Exception as e:
        err = "Unable to load ACS5 variables."
        raise Exception(err) from e


@cache
def get_group_vars(year: int, group: str) -> list[tuple[str, dict]]:
    variables = sorted(
        (
            (name, attrs)
            for name, attrs in get_vars(year).items()
            if attrs["group"] == group
        ),
        key=lambda x: x[0],
    )
    if len(variables) == 0:
        raise ValueError(f"ACS5 variable group '{group}' not found in year {year}.")
    return variables


def split_vars(acs_vars: list[str]) -> list[list[str]]:
    """
    Split a list of ACS variables so as to be suitable for querying in batches.

    The ACS API does not allow you to include more than one group per query,
    so we may need to split the list of variables into groups of non-group variables and
    group variables.
    """
    if not acs_vars:
        return []
    if acs_vars[0].startswith("group("):
        return [[acs_vars[0]], *split_vars(acs_vars[1:])]

    nongroup_vars, remaining_vars = split_at(acs_vars, lambda x: x.startswith("group("))
    return [nongroup_vars, *split_vars(remaining_vars)]


def make_queries(scope: CensusScope) -> list[dict[str, str]]:
    return acs5.ACS5Client.make_queries(scope)  # TODO


def fetch(
    scope: CensusScope,
    acs_vars: list[str],
    value_dtype: type[np.generic],
    *,
    report_progress: Callable[[float], None] | None = None,
    result_format: Literal["long", "wide"] = "long",
) -> pd.DataFrame:
    """
    Request `variables` from the Census API for the given `scope`.

    Parameters
    ----------
    scope :
        The geo scope to query.
    acs_vars :
        The list of variables to query.
    value_dtype :
        The dtype of the result array.
    report_progress :
        A callback for reporting query progress; especially useful when the scope
        necessitates multiple queries.

    Returns
    -------
    :
        A dataframe in "long" format, with columns: geoid, variable, and value.
        Geoid and variable are strings and value will be converted to the given
        dtype.
    """
    acs5_url = url(scope.year)
    var_queries = split_vars(acs_vars)
    geo_queries = make_queries(scope)
    queries = list(itertools.product(var_queries, geo_queries))
    processing_steps = len(queries) + 1
    est_var_re = re.compile(r".*?_\d\d\dE$")

    def single_query(
        query_index: int,
        session: requests.Session,
        query_vars: list[str],
        query_geo: dict[str, str],
    ) -> Iterator[list[str]]:
        try:
            response = session.get(
                acs5_url,
                params={
                    "key": census_api_key(),
                    "get": ",".join(["GEO_ID", *query_vars]),
                    **query_geo,
                },
                timeout=60,
            )
            if response.status_code == 414:
                err = (
                    "the attempted request URI was too long to send. "
                    "The root cause for this can vary, but it usually suggests "
                    "your query involves too many locations."
                )
                raise DeferredADRIOError(ADRIOCommunicationError, err)
            response.raise_for_status()
            [cols, *rows] = response.json()

            # yield long-format records, dropping non-estimate columns
            est_cols = [(i, x) for i, x in enumerate(cols) if est_var_re.match(x)]
            for row in rows:
                for col_index, col in est_cols:
                    # drop ucgid prefix from geoid, e.g., "0500000US"
                    geoid = row[0][9:]
                    yield [geoid, col, row[col_index]]

            if report_progress:
                report_progress((query_index + 1) / processing_steps)
        except (DeferredADRIOError, ADRIOError):
            raise
        except Exception as e:
            err = "unexpected error fetching ACS5 data."
            raise DeferredADRIOError(ADRIOCommunicationError, err) from e

    with requests.Session() as session:
        records = itertools.chain.from_iterable(
            single_query(i, session, query_vars, query_geo)
            for i, (query_vars, query_geo) in enumerate(queries)
        )
        result_df = pd.DataFrame.from_records(
            data=records,
            index=["geoid", "variable"],
            columns=["geoid", "variable", "value"],
        )
        result_df["value"] = result_df["value"].astype(value_dtype)
        result_df = result_df.sort_index()
        if result_format == "long":
            return result_df
        else:
            return cast(pd.DataFrame, result_df["value"].unstack(level="variable"))


_exact_pattern = re.compile(r"^(\d+) years$")
_under_pattern = re.compile(r"^Under (\d+) years$")
_range_pattern = re.compile(r"^(\d+) (?:to|and) (\d+) years")
_over_pattern = re.compile(r"^(\d+) years and over")


class AgeRange(NamedTuple):
    """
    Models an age range for use with ACS age-categorized data.
    Unlike Python integer ranges, the `end` of the this range is inclusive.
    `end` can also be None which models the "and over" part of ranges
    like "85 years and over".
    """

    start: int
    """The youngest age included in the range."""
    end: int | None
    """The oldest age included in the range, or None to indicate an unbounded range."""

    def contains(self, other: "AgeRange") -> bool:
        """
        Check if `other` range is fully contained in (or coincident with) this range.

        Parameters
        ----------
        other :
            The other age range to consider.

        Returns
        -------
        :
            True if the range is contained in this range.
        """
        if self.start > other.start:
            return False
        if self.end is None:
            return True
        if other.end is None:
            return False
        return self.end >= other.end

    @staticmethod
    def parse(label: str) -> "AgeRange | None":
        """
        Parse the age range of an ACS field label.

        For example: `Estimate!!Total:!!Male:!!22 to 24 years`.

        Parameters
        ----------
        label :
            A census variable label.

        Returns
        -------
        :
            The `AgeRange` object if parsing is successful, `None` if not.
        """
        parts = label.split("!!")
        if len(parts) != 4:
            return None
        bracket = parts[-1]
        if (m := _exact_pattern.match(bracket)) is not None:
            start = int(m.group(1))
            end = start
        elif (m := _under_pattern.match(bracket)) is not None:
            start = 0
            end = int(m.group(1)) - 1
        elif (m := _range_pattern.match(bracket)) is not None:
            start = int(m.group(1))
            end = int(m.group(2))
        elif (m := _over_pattern.match(bracket)) is not None:
            start = int(m.group(1))
            end = None
        else:
            raise ValueError(f"No match for {label}")
        return AgeRange(start, end)


AgeRangeLike = tuple[int, int | None] | AgeRange


def make_age_masks(
    scope: CensusScope,
    from_group: str,
    age_ranges: list[AgeRange],
) -> NDArray[np.int64]:
    # NOTE: we can't use the age_ranges() static method here because it omits
    # total and subtotal vars and we need to account for the whole group.
    age_vars = [
        AgeRange.parse(attrs["label"])
        for _, attrs in get_group_vars(scope.year, from_group)
    ]

    strata_masks = []
    for adrio_range in age_ranges:

        def is_included(x: AgeRange | None) -> TypeGuard[AgeRange]:
            return x is not None and adrio_range.contains(x)

        included, col_mask = filter_with_mask(age_vars, is_included)

        # At least one var must have its start equal to the ADRIO range
        if not any((x.start == adrio_range.start for x in included)):
            raise DeferredADRIOError(ADRIOProcessingError, f"bad start {adrio_range}")
        # At least one var must have its end equal to the ADRIO range
        if not any((x.end == adrio_range.end for x in included)):
            raise DeferredADRIOError(ADRIOProcessingError, f"bad end {adrio_range}")

        strata_masks.append(col_mask)
    return np.asarray(strata_masks, dtype=np.int64)


###################
# ADRIO Utilities #
###################


def validate_context(adrio: ADRIO, context: Context) -> None:
    if census_api_key() is None:
        err = (
            "Census API key is required for accessing ACS5 data. "
            "Please set the environment variable 'CENSUS_API_KEY'"
        )
        raise ADRIOContextError(adrio, context, err)
    try:
        scope = context.scope  # scope is required
    except MissingContextError as e:
        raise ADRIOContextError(adrio, context, str(e))
    if not isinstance(scope, CensusScope):
        err = "US Census geo scope required."
        raise ADRIOContextError(adrio, context, err)
    if scope.year not in ACS5_YEARS:
        err = f"{scope.year} is not a supported year for ACS5 data."
        raise ADRIOContextError(adrio, context, err)
    if isinstance(scope, BlockGroupScope) and scope.year <= 2012:
        err = "Block group ACS5 data is not available via this API for 2012 or prior."
        raise ADRIOContextError(adrio, context, err)


##########
# ADRIOs #
##########


@adrio_cache
class Population(ADRIO[np.int64, np.int64]):
    strata: list[str] | None
    age_ranges: list[AgeRange]

    fix_insufficient_data: int | None
    """
    A number to replace values that were not reported by the Census due to an
    insufficient number of sample observations (-666666666 in the data).
    """
    fix_missing: int | None
    """
    A number to replace values that are missing from the data.
    """

    def __init__(
        self,
        strata: dict[str, AgeRangeLike] | list[AgeRangeLike] | None = None,
        *,
        fix_insufficient_data: int | None = None,
        fix_missing: int | None = None,
    ):
        self.fix_insufficient_data = fix_insufficient_data
        self.fix_missing = fix_missing

        if strata is None:
            strata = {DEFAULT_STRATA: AgeRange(0, None)}

        self.strata = list(strata.keys()) if isinstance(strata, dict) else None
        strata_vals = list(strata.values()) if isinstance(strata, dict) else strata
        self.age_ranges = [
            AgeRange(*x) if isinstance(x, tuple) else x for x in strata_vals
        ]

    @property
    @override
    def result_format(self) -> ResultFormat:
        # This actually describes the shape of each stratum's results...
        return ResultFormat(shape=Shapes.N, dtype=np.int64)

    @override
    def validate_context(self, context: Context) -> None:
        return validate_context(self, context)

    @override
    def validate_result(self, context: Context, result: AttributeData) -> None:
        # At validation, result is just a numpy array where the first axis is strata.
        # TODO: I think PipelineResult should allow StratifiedAttributeArray values...
        for stratum in cast(np.ndarray, result):
            adrio_validate_pipe(
                self,
                context,
                stratum,
                validate_numpy(),
                validate_shape(self.result_format.shape.to_tuple(context.dim)),
                validate_dtype(self.result_format.dtype),
                validate_values_in_range(0, None),
            )

    @override
    def inspect(self) -> InspectResult[np.int64, np.int64]:
        context = self.context
        with adrio_exception_handling(self, context):
            scope = cast(CensusScope, context.scope)
            var_group = "B01001"

            self.validate_context(context)

            self._report_progress(0.0)
            start_time = perf_counter()

            source_df = fetch(
                scope=scope,
                acs_vars=[f"group({var_group})"],
                value_dtype=self.result_format.dtype.type,
                report_progress=self._report_progress,
                result_format="wide",
            )

            proc_res = (
                DataIssues(data=source_df)
                .fix(
                    issue_name="insufficient_data",
                    where=lambda df: df == -666666666,
                    replacement=None,
                )
                .fix(
                    issue_name="missing",
                    where=lambda df: df.isna(),
                    replacement=None,
                )
                .to_result()
            )

            if not proc_res.issues:
                # Length variables:
                # - M is number of strata,
                # - N is number of locations,
                # - V is the number of variables.
                #
                # Shapes:
                # - `Mask` is: MxV
                # - `Table` is: NxV
                # - So `Mask @ Table.T` is: MxN
                mask_matrix = make_age_masks(scope, var_group, self.age_ranges)
                result = np.matmul(mask_matrix, source_df.to_numpy().T)
                proc_res = PipelineResult(value=result, issues={})

            # At validation, result is just a numpy array where the first axis is strata.
            # TODO: maybe PipelineResult should allow StratifiedAttributeArray values...
            # or maybe we should bypass it entirely...
            result_np = proc_res.value_as_masked
            self.validate_result(context, result_np)

            finish_time = perf_counter()
            self._report_complete(finish_time - start_time)

            return InspectResult(
                self,
                source_df,
                result_np,
                self.result_format.dtype.type,
                self.result_format.shape,
                proc_res.issues,
            )

    @override
    def evaluate(self) -> AttributeData:
        # TODO: coercing to StratifiedAttributeArray here is a bit of a hack
        # because InspectResult doesn't really support it
        return StratifiedAttributeArray(
            values=cast(np.ndarray, super().evaluate()),
            strata=self.strata,
        )
