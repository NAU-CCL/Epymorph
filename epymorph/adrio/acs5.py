"""ADRIOs that access the US Census ACS 5-year data."""

import re
from collections import defaultdict
from functools import cache
from json import load as load_json
from os import environ
from typing import Literal, NamedTuple, Sequence, TypeGuard

import numpy as np
import pandas as pd
from census import Census, CensusException
from numpy.typing import NDArray
from pandas import DataFrame
from typing_extensions import override

from epymorph.adrio.adrio import Adrio, ProgressCallback, adrio_cache
from epymorph.attribute import AttributeDef
from epymorph.cache import load_or_fetch_url, module_cache_path
from epymorph.data_shape import Shapes
from epymorph.error import DataResourceException
from epymorph.geography.scope import GeoScope
from epymorph.geography.us_census import (
    BlockGroupScope,
    CensusScope,
    CountyScope,
    StateScope,
    TractScope,
)
from epymorph.geography.us_geography import (
    BLOCK_GROUP,
    COUNTY,
    STATE,
    TRACT,
    CensusGranularity,
)
from epymorph.util import filter_with_mask

_ACS5_CACHE_PATH = module_cache_path(__name__)

Acs5Year = Literal[
    2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022
]
"""A supported ACS5 data year."""

ACS5_YEARS: Sequence[Acs5Year] = (
    2009,
    2010,
    2011,
    2012,
    2013,
    2014,
    2015,
    2016,
    2017,
    2018,
    2019,
    2020,
    2021,
    2022,
)
"""All supported ACS5 data years."""


@cache
def _get_api() -> Census:
    api_key = environ.get("CENSUS_API_KEY")
    if api_key is None:
        msg = (
            "Census API key not found. "
            "Please ensure you have set the environment variable 'CENSUS_API_KEY'"
        )
        raise DataResourceException(msg)
    return Census(api_key)


@cache
def _get_vars(year: int) -> dict[str, dict]:
    try:
        vars_url = f"https://api.census.gov/data/{year}/acs/acs5/variables.json"
        cache_path = _ACS5_CACHE_PATH / f"variables-{year}.json"
        file = load_or_fetch_url(vars_url, cache_path)
        return load_json(file)["variables"]
    except Exception as e:
        raise DataResourceException("Unable to load ACS5 variables.") from e


@cache
def _get_group_vars(year: int, group: str) -> list[tuple[str, dict]]:
    return sorted(
        (
            (name, attrs)
            for name, attrs in _get_vars(year).items()
            if attrs["group"] == group
        ),
        key=lambda x: x[0],
    )


def _validate_scope(scope: GeoScope) -> CensusScope:
    if not isinstance(scope, CensusScope):
        raise DataResourceException("Census scope is required for acs5 attributes.")
    if not is_acs5_year(scope.year):
        raise DataResourceException(
            f"{scope.year} is not a supported year for acs5 attributes."
        )
    return scope


def is_acs5_year(year: int) -> TypeGuard[Acs5Year]:
    """A type-guard function to ensure a year is a supported ACS5 year."""
    return year in ACS5_YEARS


def _make_acs5_queries(scope: CensusScope) -> list[dict[str, str]]:
    """Creates census API queries that cover the given scope."""
    match scope:
        case StateScope(includes_granularity="state", includes=includes):
            if scope.is_all_states():
                return [{"for": "state:*"}]
            else:
                return [{"for": f"state:{','.join(includes)}"}]

        case CountyScope(includes_granularity="state", includes=includes):
            return [
                {
                    "for": "county:*",
                    "in": f"state:{','.join(includes)}",
                }
            ]

        case CountyScope(includes_granularity="county", includes=includes):
            counties_by_state: dict[str, list[str]] = defaultdict(list)
            for state, county in map(COUNTY.decompose, includes):
                counties_by_state[state].append(county)
            return [
                {"for": f"county:{','.join(cs)}", "in": f"state:{s}"}
                for s, cs in counties_by_state.items()
            ]

        case TractScope(includes_granularity="state", includes=includes):
            return [
                {
                    "for": "tract:*",
                    "in": f"state:{','.join(includes)} county:*",
                }
            ]

        case TractScope(includes_granularity="county", includes=includes):
            counties_by_state: dict[str, list[str]] = defaultdict(list)
            for state, county in map(COUNTY.decompose, includes):
                counties_by_state[state].append(county)
            return [
                {"for": "tract:*", "in": f"state:{s} county:{','.join(cs)}"}
                for s, cs in counties_by_state.items()
            ]

        case TractScope(includes_granularity="tract", includes=includes):
            tracts_by_county: dict[str, list[str]] = defaultdict(list)

            for state, county, tract in map(TRACT.decompose, includes):
                tracts_by_county[state + county].append(tract)

            return [
                {
                    "for": f"tract:{','.join(tracts_by_county[state + county])}",
                    "in": f"state:{state} county:{county}",
                }
                for state, county in [
                    COUNTY.decompose(c) for c in tracts_by_county.keys()
                ]
            ]

        case BlockGroupScope(includes_granularity="state", includes=includes):
            # This wouldn't normally need to be multiple queries,
            # but Census API won't let you fetch CBGs for multiple states.
            states = {STATE.extract(x) for x in includes}
            return [
                {"for": "block group:*", "in": f"state:{s} county:* tract:*"}
                for s in states
            ]

        case BlockGroupScope(includes_granularity="county", includes=includes):
            counties_by_state: dict[str, list[str]] = defaultdict(list)
            for state, county in map(COUNTY.decompose, includes):
                counties_by_state[state].append(county)
            return [
                {
                    "for": "block group:*",
                    "in": f"state:{s} county:{','.join(cs)} tract:*",
                }
                for s, cs in counties_by_state.items()
            ]

        case BlockGroupScope(includes_granularity="tract", includes=includes):
            tracts_by_county: dict[str, list[str]] = defaultdict(list)

            for state, county, tract in map(TRACT.decompose, includes):
                tracts_by_county[state + county].append(tract)

            return [
                {
                    "for": "block group:*",
                    "in": (
                        f"state:{state} county:{county} "
                        f"tract:{','.join(tracts_by_county[state + county])}"
                    ),
                }
                for state, county in [
                    COUNTY.decompose(c) for c in tracts_by_county.keys()
                ]
            ]

        case BlockGroupScope(includes_granularity="block group", includes=includes):
            block_groups_by_tract: dict[str, list[str]] = defaultdict(list)

            for state, county, tract, block_group in map(
                BLOCK_GROUP.decompose, includes
            ):
                block_groups_by_tract[state + county + tract].append(block_group)

            return [
                {
                    "for": f"block group:{'.'.join(block_groups_by_tract[state + county + tract])}",  # noqa: E501
                    "in": f"state:{state} county:{county} tract:{tract}",
                }
                for state, county, tract in [
                    TRACT.decompose(t) for t in block_groups_by_tract.keys()
                ]
            ]

        case _:
            raise DataResourceException("Unsupported query.")


def _fetch_acs5(
    variables: list[str], scope: CensusScope, progress: ProgressCallback
) -> DataFrame:
    census = _get_api()
    queries = _make_acs5_queries(scope)

    # fetch all queries and combine results
    try:
        acs_data = list[DataFrame]()
        processing_steps = len(queries) + 1
        for i, query in enumerate(queries):
            acs_data.append(
                pd.DataFrame.from_records(
                    census.acs5.get(variables, geo=query, year=scope.year)
                )
            )

            progress((i + 1) / processing_steps, None)

        acs_df = pd.concat(acs_data)

    except CensusException as e:
        err = "Unable to load data from the US Census API at this time."
        raise DataResourceException(err) from e
    if acs_df.empty:
        msg = (
            "ACS5 query returned empty. "
            "Ensure all geographies included in your scope are supported and try again."
        )
        raise DataResourceException(msg)

    # concatenate geoid components to create 'geoid' column
    columns: list[str] = {
        "state": ["state"],
        "county": ["state", "county"],
        "tract": ["state", "county", "tract"],
        "block group": ["state", "county", "tract", "block group"],
    }[scope.granularity]
    acs_df["geoid"] = acs_df[columns].apply("".join, axis=1)

    # check and sort results for 1:1 match with scope
    try:
        geoid_df = pd.DataFrame({"geoid": scope.node_ids})
        return geoid_df.merge(acs_df, on="geoid", how="left", validate="1:1")
    except pd.errors.MergeError:
        msg = "Fetched data was not an exact match for the scope's geographies."
        raise DataResourceException(msg) from None


@adrio_cache
class Population(Adrio[np.int64]):
    """
    Retrieves an N-shaped array of integers representing the total population of each
    geographic node. Data is retrieved from Census table variable B01001_001 using ACS5
    5-year estimates.
    """

    @override
    def evaluate_adrio(self) -> NDArray[np.int64]:
        scope = _validate_scope(self.scope)
        acs_df = _fetch_acs5(["B01001_001E"], scope, self.progress)
        return acs_df["B01001_001E"].to_numpy(dtype=np.int64)


@adrio_cache
class PopulationByAgeTable(Adrio[np.int64]):
    """
    Creates a table of population data for each geographic node split into various age
    categories. Data is retrieved from Census table B01001 using ACS5 5-year estimates.
    """

    @override
    def evaluate_adrio(self) -> NDArray[np.int64]:
        scope = _validate_scope(self.scope)
        # NOTE: asking acs5 explicitly for the [B01001_001E, ...] vars
        # seems to be about twice as fast as asking for group(B01001)
        age_vars = [var for var, _ in _get_group_vars(scope.year, "B01001")]
        age_vars.sort()
        acs_df = _fetch_acs5(age_vars, scope, self.progress)
        return acs_df[age_vars].to_numpy(dtype=np.int64)


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
    end: int | None

    def contains(self, other: "AgeRange") -> bool:
        """Is the `other` range fully contained in (or coincident with) this range?"""
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
        Parse the age range of an ACS field label;
        e.g.: `Estimate!!Total:!!Male:!!22 to 24 years`
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
            end = int(m.group(1))
        elif (m := _range_pattern.match(bracket)) is not None:
            start = int(m.group(1))
            end = int(m.group(2))
        elif (m := _over_pattern.match(bracket)) is not None:
            start = int(m.group(1))
            end = None
        else:
            raise DataResourceException(f"No match for {label}")
        return AgeRange(start, end)


@adrio_cache
class PopulationByAge(Adrio[np.int64]):
    """
    Retrieves an N-shaped array of integers representing the total population
    within a specified age range for each geographic node. Data is retrieved from
    a population by age table constructed from Census table B01001.
    """

    POP_BY_AGE_TABLE = AttributeDef("population_by_age_table", int, Shapes.NxA)

    requirements = [POP_BY_AGE_TABLE]
    """The population by age table is required for retrieving the population totals by
    age range."""

    _age_range: AgeRange

    def __init__(self, age_range_start: int, age_range_end: int | None):
        """
        Initializes the population by age array with a starting age and an end age to
        define the total age range.

        Parameters
        ----------
        age_range_start : int
            The starting age for the age range to retrieve from the population table.
        age_range_end : int | None
            The inclusive ending age for the age range to retrieve from the population
            table. If None, the range includes all ages at and above age_range_start.
        """
        self._age_range = AgeRange(age_range_start, age_range_end)

    @override
    def evaluate_adrio(self) -> NDArray[np.int64]:
        scope = _validate_scope(self.scope)

        age_ranges = [
            AgeRange.parse(attrs["label"])
            for var, attrs in _get_group_vars(scope.year, "B01001")
        ]

        adrio_range = self._age_range

        def is_included(x: AgeRange | None) -> TypeGuard[AgeRange]:
            return x is not None and adrio_range.contains(x)

        included, col_mask = filter_with_mask(age_ranges, is_included)

        # At least one var must have its start equal to the ADRIO range
        if not any((x.start == adrio_range.start for x in included)):
            raise DataResourceException(f"bad start {adrio_range}")
        # At least one var must have its end equal to the ADRIO range
        if not any((x.end == adrio_range.end for x in included)):
            raise DataResourceException(f"bad end {adrio_range}")

        table = self.data(self.POP_BY_AGE_TABLE)
        return table[:, col_mask].sum(axis=1)


@adrio_cache
class AverageHouseholdSize(Adrio[np.float64]):
    """
    Retrieves an N-shaped array of floats representing the average number of people
    living in each household for every geographic node.
    Data is retrieved from Census table variable B25010_001 using ACS5 5-year estimates.
    """

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        scope = _validate_scope(self.scope)
        acs_df = _fetch_acs5(["B25010_001E"], scope, self.progress)
        return acs_df["B25010_001E"].to_numpy(dtype=np.float64)


@adrio_cache
class DissimilarityIndex(Adrio[np.float64]):
    """
    Calculates an N-shaped array of floats representing the amount of racial segregation
    between a specified racial majority and minority groups on a scale of
    0 (complete integration) to 1 (complete segregation). Data is calculated using
    population data from Census table B02001 using ACS5 5-year estimates.
    """

    RaceCategory = Literal[
        "White", "Black", "Native", "Asian", "Pacific Islander", "Other"
    ]

    race_variables: dict[RaceCategory, str] = {
        "White": "B02001_002E",
        "Black": "B02001_003E",
        "Native": "B02001_004E",
        "Asian": "B02001_005E",
        "Pacific Islander": "B02001_006E",
        "Other": "B02001_007E",
    }

    majority_pop: RaceCategory
    minority_pop: RaceCategory

    def __init__(self, majority_pop: RaceCategory, minority_pop: RaceCategory):
        """
        Initializes the fetching for the array of segregation with a majority and
        minority population.

        Parameters
        ----------
        majority_pop : RaceCategory
            The race category representing the majority population for the amount of
            segregation (options:, 'White', 'Black', 'Native', 'Asian',
            'Pacific Islander', 'Other').

        minority_pop : RaceCategory
            The race category representing the minority population within the
            segregation analysis (options:, 'White', 'Black', 'Native', 'Asian',
            'Pacific Islander', 'Other').
        """
        self.majority_pop = majority_pop
        """The race category of the majority population"""
        self.minority_pop = minority_pop
        """The race category of the minority population of interest"""

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        high_scope = _validate_scope(self.scope)
        if isinstance(high_scope, BlockGroupScope):
            msg = "Dissimilarity index cannot be retreived for block group scope."
            raise DataResourceException(msg)
        low_scope = high_scope.lower_granularity()

        majority_var = self.race_variables[self.majority_pop]
        minority_var = self.race_variables[self.minority_pop]

        # Fetch the data for the upper scope.
        high_df = _fetch_acs5(
            [majority_var, minority_var], high_scope, self.progress
        ).rename(columns={majority_var: "high_majority", minority_var: "high_minority"})
        # Fetch the data for the lower scope.
        low_df = _fetch_acs5(
            [majority_var, minority_var], low_scope, self.progress
        ).rename(columns={majority_var: "low_majority", minority_var: "low_minority"})
        # Create a merge key by truncating the lower scope's GEOID.
        gran = CensusGranularity.of(high_scope.granularity)
        low_df["geoid"] = low_df["geoid"].apply(gran.truncate)

        # Merge and compute score.
        result_df = high_df.merge(low_df, on="geoid")
        result_df["score"] = abs(
            result_df["low_minority"] / result_df["high_minority"]
            - result_df["low_majority"] / result_df["high_majority"]
        )
        result_df = result_df.groupby("geoid").sum()
        result_df["score"] *= 0.5
        result_df["score"] = result_df["score"].replace(0.0, 0.5)
        result_df = result_df.reset_index()
        return result_df["score"].to_numpy(dtype=np.float64)


@adrio_cache
class GiniIndex(Adrio[np.float64]):
    """
    Retrieves an N-shaped array of floats representing the amount of income inequality
    on a scale of 0 (perfect equality) to 1 (perfect inequality) for each
    geographic node. Data is retrieved from Census table variable B19083_001 using
    ACS 5-year estimates.
    """

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        scope = _validate_scope(self.scope)

        # if granularity is block groups, use the data of the parent tract
        if isinstance(scope, BlockGroupScope):
            print(
                "Gini Index cannot be retrieved for block group level, "
                "fetching tract level data instead."
            )
            tracts = scope.raise_granularity()
            tract_df = _fetch_acs5(["B19083_001E"], tracts, self.progress)
            as_tract = np.vectorize(TRACT.truncate)
            cbg_df = pd.DataFrame(
                {
                    "geoid": scope.node_ids,
                    "tract_geoid": as_tract(scope.node_ids),
                }
            )
            result_df = cbg_df.merge(
                tract_df,
                left_on="tract_geoid",
                right_on="geoid",
                suffixes=(None, "_y"),
            )
        else:
            result_df = _fetch_acs5(["B19083_001E"], scope, self.progress)

        return (
            result_df["B19083_001E"]
            .astype(np.float64)
            .fillna(0.5)
            .replace(-666666666, 0.5)
            .to_numpy(dtype=np.float64)
        )


@adrio_cache
class MedianAge(Adrio[np.float64]):
    """
    Retrieves an N-shaped array of floats representing the median age in
    each geographic node. Data is retrieved from Census table variable
    B01002_001 using ACS 5-year estimates.
    """

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        scope = _validate_scope(self.scope)
        acs_df = _fetch_acs5(["B01002_001E"], scope, self.progress)
        return acs_df["B01002_001E"].to_numpy(dtype=np.float64)


@adrio_cache
class MedianIncome(Adrio[np.float64]):
    """
    Retrieves an N-shaped array of floats representing the median yearly income in
    each geographic node. Data is retrieved from Census table variable B19013_001
    using ACS 5-year estimates.
    """

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        scope = _validate_scope(self.scope)
        acs_df = _fetch_acs5(["B19013_001E"], scope, self.progress)
        return acs_df["B19013_001E"].to_numpy(dtype=np.float64)
