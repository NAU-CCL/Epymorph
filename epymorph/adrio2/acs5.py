import dataclasses
import itertools
import json
import os
import re
from collections.abc import Collection, Iterator, Mapping, Sequence
from functools import cache
from typing import Any, Literal, NamedTuple, NotRequired, TypedDict

import numpy as np
import pandas as pd
import requests
from numpy.typing import NDArray
from typing_extensions import override

import epymorph.adrio2.validate as v
from epymorph.adrio import acs5
from epymorph.adrio2.adrio import (
    ADRIO,
    ADRIOCommunicationError,
    ADRIOContextError,
    ADRIOError,
    DeferredADRIOError,
    InspectResult,
    ProgressCallback,
    adrio_cache,
    adrio_exception_handling,
    adrio_progress,
)
from epymorph.cache import load_or_fetch_url, module_cache_path
from epymorph.data_shape import Shapes
from epymorph.data_type import AttributeData, StratifiedAttributeArray
from epymorph.error import MissingContextError
from epymorph.geography.us_census import BlockGroupScope, CensusScope
from epymorph.simulation import Context
from epymorph.strata import DEFAULT_STRATA
from epymorph.util import split_at

######################
# ACS5 API Functions #
######################

# fmt:off
ACS5_YEARS: Sequence[int] = (2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024)  # noqa: E501
"""All supported ACS5 data years."""

RaceCategory = Literal[
    "White Alone", "Black or African American Alone",
    "American Indian and Alaska Native Alone", "Asian Alone",
    "Native Hawaiian and Other Pacific Islander Alone", "Some Other Race Alone",
    "Two or More Races"
]
"""A racial category defined by ACS5."""

_RACE_VARIABLES: Mapping[RaceCategory, str] = {
    "White Alone": "B01001A",
    "Black or African American Alone": "B01001B",
    "American Indian and Alaska Native Alone": "B01001C",
    "Asian Alone": "B01001D",
    "Native Hawaiian and Other Pacific Islander Alone": "B01001E",
    "Some Other Race Alone": "B01001F",
    "Two or More Races": "B01001G",
}

SexCategory = Literal["Male", "Female"]
"""A sex category defined by ACS5."""
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
def get_var_attrs(year: int, var_name: str) -> dict:
    try:
        return get_vars(year)[var_name]
    except KeyError:
        err = f"ACS5 variable '{var_name}' not found in year {year}."
        raise ValueError(err)


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
    return acs5.ACS5Client.make_queries(scope)  # TODO: migrate here


def fetch(
    scope: CensusScope,
    acs_vars: list[str],
    value_dtype: type[np.generic],
    *,
    report_progress: ProgressCallback | None = None,
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
        return result_df["value"].unstack(level="variable")


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

    def __str__(self) -> str:
        if self.end is None:
            return f"{self.start}+"
        if self.start == self.end:
            return str(self.start)
        return f"{self.start}-{self.end}"

    @override
    def __contains__(self, other: object) -> bool:
        if not isinstance(other, AgeRange):
            return False
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

    @staticmethod
    def coerce(value: "AgeRangeLike") -> "AgeRange":
        if isinstance(value, AgeRange):
            return value
        start, end = value
        return AgeRange(start, end)


AgeRangeLike = tuple[int, int | None] | AgeRange


###################
# ADRIO Utilities #
###################


def validate_context(adrio: ADRIO, context: Context) -> CensusScope:
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
    return scope


class PopulationStrata(TypedDict):
    age: NotRequired[dict[str, AgeRangeLike] | list[AgeRangeLike]]
    race: NotRequired[dict[str, RaceCategory] | list[RaceCategory]]
    sex: NotRequired[dict[str, SexCategory] | list[SexCategory]]


class PopulationStrataConfig(TypedDict):
    age: NotRequired[dict[str, AgeRange]]
    race: NotRequired[dict[str, RaceCategory]]
    sex: NotRequired[dict[str, SexCategory]]


def normalize_strata(
    strata: PopulationStrata | PopulationStrataConfig | None,
) -> PopulationStrataConfig:
    normalized: PopulationStrataConfig = {}
    if strata is None:
        return normalized
    for category, category_spec in strata.items():
        if not category_spec:
            continue  # Treat empty dicts or lists as if the category is unspecified

        if category == "age":
            if isinstance(category_spec, dict):
                normalized["age"] = {
                    str(name): AgeRange.coerce(value)
                    for name, value in category_spec.items()
                }
            elif isinstance(category_spec, list):
                ages = [AgeRange.coerce(value) for value in category_spec]
                normalized["age"] = {str(age): age for age in ages}

        elif category == "race":
            if isinstance(category_spec, dict):
                normalized["race"] = category_spec
            elif isinstance(category_spec, list):
                normalized["race"] = {str(value): value for value in category_spec}

        elif category == "sex":
            if isinstance(category_spec, dict):
                normalized["sex"] = category_spec
            elif isinstance(category_spec, list):
                normalized["sex"] = {str(value): value for value in category_spec}
    return normalized


@dataclasses.dataclass(frozen=True)
class Stratum:
    name: str
    age: AgeRange | None
    race: RaceCategory | None
    sex: SexCategory | None


_AxisOption = tuple[str, dict[str, Any]]
"""A single item from a PopulationStrataConfig dict."""


def iter_strata(strata: PopulationStrataConfig | None) -> Iterator[Stratum]:
    if not strata:
        yield Stratum(DEFAULT_STRATA, None, None, None)
        return

    def recurse(curr: Stratum, axis_options: list[_AxisOption]) -> Iterator[Stratum]:
        if not axis_options:
            yield curr
            return

        [(axis, options), *rest] = axis_options
        for name, value in options.items():
            changes = {
                "name": f"{curr.name}_{name}" if curr.name else name,
                axis: value,
            }
            next_stratum = dataclasses.replace(curr, **changes)
            yield from recurse(next_stratum, rest)

    yield from recurse(
        Stratum("", None, None, None),
        axis_options=list(strata.items()),  # pyright: ignore[reportArgumentType]
    )


def population_vars(strata: PopulationStrataConfig) -> list[str]:
    match (strata.get("age"), strata.get("sex"), strata.get("race")):
        case (None, None, None):  # Total population
            return ["B01001_001E"]
        case (_, None, None):  # Age only
            return ["group(B01001)"]
        case (None, _, None):  # Sex only
            return ["B01001_002E", "B01001_026E"]
        case (_, _, None):  # Age and Sex
            return ["group(B01001)"]
        case (_, _, race_strata):  # Any combination involving Race
            return [f"group({_RACE_VARIABLES[cat]})" for cat in race_strata.values()]


def population_var_in_stratum(stratum: Stratum, acs_var: str, acs_year: int) -> bool:
    attrs = get_var_attrs(acs_year, acs_var)
    label = attrs.get("label", "")
    parts = label.split("!!")

    # A single data table (group) includes disaggregation by:
    #   - total (no breakdown)
    #   - sex
    #   - sex and age
    # We can use only the top-line total if both age and sex are unspecified.
    # We can use only the sex breakdown if age is unspecified but sex is specified.
    # Otherwise, we must use the sex and age breakdowns.
    # If age is specified but sex is not, include male and female (summed) by age.
    # Race disaggregation is handled as separate groups,
    # so if race is specified, only use variables from that race's group.

    var_age = AgeRange.parse(label)
    if stratum.age is None:
        if var_age is not None:
            return False
    elif var_age not in stratum.age:
        return False

    var_sex = "Male" if "Male:" in parts else "Female" if "Female:" in parts else None
    if stratum.sex is None:
        if stratum.age is not None and var_sex is None:
            return False  # ignore totals
        if stratum.age is None and var_sex is not None:
            return False  # ignore sex breakdowns
    elif stratum.sex != var_sex:
        return False

    if stratum.race is not None:
        table = acs_var.split("_")[0]
        var_race = None
        for k, v in _RACE_VARIABLES.items():
            if table == v:
                var_race = k
                break
        if var_race != stratum.race:
            return False
    return True


def population_mask(
    strata: PopulationStrataConfig,
    acs_vars: Collection[str],
    acs_year: int,
) -> NDArray[np.bool_]:
    return np.array(
        [
            [
                population_var_in_stratum(stratum, var_name, acs_year)
                for var_name in acs_vars
            ]
            for stratum in iter_strata(strata)
        ],
        dtype=np.bool_,
    )


##########
# ADRIOs #
##########


@adrio_cache
class Population(ADRIO):
    strata: list[str] | None
    strata_config: PopulationStrataConfig

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
        strata: PopulationStrata | PopulationStrataConfig | None = None,
        *,
        fix_insufficient_data: int | None = None,
        fix_missing: int | None = None,
    ):
        self.fix_insufficient_data = fix_insufficient_data
        self.fix_missing = fix_missing

        self.strata_config = normalize_strata(strata)
        self.strata = [stratum.name for stratum in iter_strata(self.strata_config)]

    @property
    @override
    def result_format(self) -> v.ResultFormat:
        # This actually describes the shape of each stratum's results...
        return v.ResultFormat(shape=Shapes.N, dtype=np.int64)

    @override
    def inspect(self) -> InspectResult:
        with (
            adrio_exception_handling(self, self.context),
            adrio_progress(self) as report_progress,
        ):
            strata = self.strata_config
            scope = validate_context(self, self.context)

            # TODO: validate age ranges...
            # age_ranges = [AgeRange.parse(attrs["label"]) for _, attrs in reference_vars]
            # for adrio_range in (strata.get("age") or {}).values():
            #     included = [age for age in age_ranges if age is not None and age in adrio_range]
            #     if not any(age.start == adrio_range.start for age in included):
            #         raise DeferredADRIOError(ADRIOProcessingError, f"bad start {adrio_range}")
            #     if not any(age.end == adrio_range.end for age in included):
            #         raise DeferredADRIOError(ADRIOProcessingError, f"bad end {adrio_range}")

            # Convert the post-processed dataframe into a stratified result.
            def to_result(df: pd.DataFrame) -> AttributeData:
                # For M strata, N locations, and V variables:
                #  `Mask` is: (M,V)
                #  `Table` is: (N,V)
                # So `Mask @ Table.T` is: (M,N)
                mask_matrix = population_mask(strata, df.columns, scope.year)
                result_np = np.matmul(mask_matrix, df.to_numpy().T)
                # Validate result
                num_strata = len(self.strata) if self.strata is not None else 1
                exp_shape = (num_strata, *Shapes.N.to_tuple(self.context.dim))
                v.is_shape(result_np, exp_shape)
                v.is_dtype(result_np, np.int64)
                v.values_in_range(result_np, 0, None)
                return StratifiedAttributeArray(values=result_np, strata=self.strata)

            return (
                InspectResult.from_source(
                    fetch(
                        scope=scope,
                        acs_vars=population_vars(strata),
                        value_dtype=np.int64,
                        report_progress=report_progress,
                        result_format="wide",
                    )
                )
                .fix(
                    issue_name="insufficient_data",
                    where=lambda df: df == -666666666,
                    replacement=self.fix_insufficient_data,
                )
                .fix(
                    issue_name="missing",
                    where=lambda df: df.isna(),
                    replacement=self.fix_missing,
                )
                .map(to_result)
            )
