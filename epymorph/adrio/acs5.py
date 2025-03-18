import os
import re
from abc import abstractmethod
from collections import defaultdict
from functools import cache, reduce
from json import load as load_json
from typing import Callable, Literal, NamedTuple, Sequence, TypeGuard

import numpy as np
import pandas as pd
import requests
from numpy.typing import NDArray
from typing_extensions import override

from epymorph.adrio.adrio import (
    ADRIOCommunicationError,
    ADRIOContextError,
    ADRIOError,
    ADRIOProcessingError,
    ADRIOPrototype,
    FetchADRIO,
    InspectResult,
    ProcessResult,
    ResultFormat,
    ValueT,
    range_mask_fn,
)
from epymorph.adrio.processing import (
    DontFill,
    Fill,
    Fix,
    FixSentinel,
    process_n,
    process_nxa,
)
from epymorph.attribute import AttributeDef
from epymorph.cache import load_or_fetch_url, module_cache_path
from epymorph.data_shape import Shapes
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
from epymorph.simulation import Context
from epymorph.util import filter_with_mask


def census_api_key() -> str | None:
    """
    Loads the API key to use for census.gov,
    as environment variable 'API_KEY__census.gov'.
    """
    key = os.environ.get("API_KEY__census.gov", default=None)
    if key is None:
        key = os.environ.get("CENSUS_API_KEY", default=None)
    return key


# fmt:off
ACS5Year = Literal[2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]  # noqa: E501
"""A supported ACS5 data year."""

ACS5_YEARS: Sequence[ACS5Year] = (2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022)  # noqa: E501
"""All supported ACS5 data years."""
# fmt: on

_ACS5_CACHE_PATH = module_cache_path(__name__)


class ACS5Client:
    @staticmethod
    def url(year: int) -> str:
        return f"https://api.census.gov/data/{year}/acs/acs5"

    @cache
    @staticmethod
    def get_vars(year: int) -> dict[str, dict]:
        try:
            vars_url = f"{ACS5Client.url(year)}/variables.json"
            cache_path = _ACS5_CACHE_PATH / f"variables-{year}.json"
            file = load_or_fetch_url(vars_url, cache_path)
            return load_json(file)["variables"]
        except Exception as e:
            err = "Unable to load ACS5 variables."
            raise ADRIOError(err) from e

    @cache
    @staticmethod
    def get_group_vars(year: int, group: str) -> list[tuple[str, dict]]:
        return sorted(
            (
                (name, attrs)
                for name, attrs in ACS5Client.get_vars(year).items()
                if attrs["group"] == group
            ),
            key=lambda x: x[0],
        )

    @cache
    @staticmethod
    def get_group_var_names(year: int, group: str) -> list[str]:
        return [var for var, _ in ACS5Client.get_group_vars(year, group)]

    @staticmethod
    def make_queries(scope: CensusScope) -> list[dict[str, str]]:
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
                raise ADRIOError("Unsupported geo scope.")

    @staticmethod
    def fetch(
        scope: CensusScope,
        variables: list[str],
        value_dtype: type[np.generic],
        report_progress: Callable[[float], None],
    ) -> pd.DataFrame:
        url = ACS5Client.url(scope.year)
        params = {
            "key": census_api_key(),
            "get": ",".join(["GEO_ID", *variables]),
        }
        queries = ACS5Client.make_queries(scope)
        processing_steps = len(queries) + 1
        with requests.Session() as session:
            results = []
            for i, q in enumerate(queries):
                response = session.get(
                    url,
                    params={**params, **q},
                    timeout=30,
                )
                response.raise_for_status()  # Raise an error for bad status codes
                [columns, *rows] = response.json()

                # keep all estimate columns
                estimate_var = re.compile(r".*?_\d\d\dE$")
                column_sel = [i for i, x in enumerate(columns) if estimate_var.match(x)]

                # convert to "long" DataFrame
                result_df = pd.DataFrame.from_records(
                    [
                        # drop ucgid prefix from geoid, e.g., "0500000US"
                        (row[0][9:], columns[col], row[col])
                        for row in rows
                        for col in column_sel
                    ],
                    columns=["geoid", "variable", "value"],
                    index="geoid",
                )
                result_df["value"] = result_df["value"].astype(value_dtype)

                results.append(result_df)
                report_progress((i + 1) / processing_steps)

            return pd.concat(results).sort_index()


def _get_scope(adrio: ADRIOPrototype, context: Context) -> CensusScope:
    if not isinstance(context.scope, CensusScope):
        err = "US Census geo scope required."
        raise ADRIOContextError(adrio, context, err)
    if context.scope.year not in ACS5_YEARS:
        err = f"{context.scope.year} is not a supported year for ACS5 data."
        raise ADRIOContextError(adrio, context, err)
    return context.scope


class _FetchACS5(FetchADRIO[ValueT, ValueT]):
    @property
    @abstractmethod
    def _variables(self) -> list[str]:
        """The ACS variables to fetch for this ADRIO."""

    @override
    def validate_context(self, context: Context):
        _get_scope(self, context)
        if census_api_key() is None:
            err = (
                "Census API key is required for accessing ACS5 data. "
                "Please set the environment variable 'CENSUS_API_KEY'"
            )
            raise ADRIOContextError(self, context, err)

    @override
    def _fetch(self, context: Context) -> pd.DataFrame:
        try:
            return ACS5Client.fetch(
                scope=_get_scope(self, context),
                variables=self._variables,
                value_dtype=self.result_format.value_dtype,
                report_progress=self.report_progress,
            )
        except Exception as e:
            raise ADRIOCommunicationError(self, context) from e

    @override
    def _process(
        self,
        context: Context,
        data_df: pd.DataFrame,
    ) -> ProcessResult[ValueT]:
        vrbs = self._variables
        if len(vrbs) == 1:
            return process_n(
                sentinels=[],
                fix_missing=DontFill(),
                dtype=self.result_format.value_dtype,
                context=context,
                data_df=data_df,
                value_name=vrbs[0],
            )
        else:
            return process_nxa(
                sentinels=[],
                fix_missing=DontFill(),
                dtype=self.result_format.value_dtype,
                context=context,
                data_df=data_df,
                value_names=vrbs,
            )


class Population(_FetchACS5[np.int64]):
    _RESULT_FORMAT = ResultFormat(
        shape=Shapes.N,
        value_dtype=np.int64,
        validation=range_mask_fn(minimum=np.int64(0), maximum=None),
        is_date_value=False,
    )

    @property
    @override
    def result_format(self) -> ResultFormat[np.int64]:
        return Population._RESULT_FORMAT

    @property
    @override
    def _variables(self) -> list[str]:
        return ["B01001_001E"]


class PopulationByAgeTable(_FetchACS5[np.int64]):
    _RESULT_FORMAT = ResultFormat(
        shape=Shapes.NxA,
        value_dtype=np.int64,
        validation=range_mask_fn(minimum=np.int64(0), maximum=None),
        is_date_value=False,
    )

    @property
    @override
    def result_format(self) -> ResultFormat[np.int64]:
        return PopulationByAgeTable._RESULT_FORMAT

    @property
    @override
    def _variables(self) -> list[str]:
        return ["group(B01001)"]

    @override
    def _validate_result(
        self,
        context: Context,
        result: NDArray[np.int64],
        *,
        expected_shape: tuple[int, ...] | None = None,
    ) -> None:
        scope = _get_scope(self, context)
        num_vars = len(ACS5Client.get_group_var_names(scope.year, "B01001"))
        shp = (context.scope.nodes, num_vars)
        super()._validate_result(context, result, expected_shape=shp)


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
            raise ValueError(f"No match for {label}")
        return AgeRange(start, end)


class PopulationByAge(ADRIOPrototype[np.int64, np.int64]):
    _RESULT_FORMAT = ResultFormat(
        shape=Shapes.N,
        value_dtype=np.int64,
        validation=range_mask_fn(minimum=np.int64(0), maximum=None),
        is_date_value=False,
    )

    # a nice feature would be for ADRIOs to suggest defaults for their requirements,
    # so if someone doesn't provide a 'population_by_age_table' for us, we can add
    # it to the requirements resolution automatically; have to make sure this is done
    # so that it's re-usable by other matching suggestions
    # (e.g., three different PopByAge)

    POP_BY_AGE_TABLE = AttributeDef("population_by_age_table", int, Shapes.NxA)

    requirements = (POP_BY_AGE_TABLE,)

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

    @property
    @override
    def result_format(self) -> ResultFormat[np.int64]:
        return PopulationByAgeTable._RESULT_FORMAT

    @override
    def validate_context(self, context: Context) -> None:
        _get_scope(self, context)

    @override
    def inspect(self) -> InspectResult[np.int64, np.int64]:
        scope = _get_scope(self, self.context)
        age_ranges = [
            AgeRange.parse(attrs["label"])
            for var, attrs in ACS5Client.get_group_vars(scope.year, "B01001")
        ]

        adrio_range = self._age_range

        def is_included(x: AgeRange | None) -> TypeGuard[AgeRange]:
            return x is not None and adrio_range.contains(x)

        included, col_mask = filter_with_mask(age_ranges, is_included)

        # At least one var must have its start equal to the ADRIO range
        if not any((x.start == adrio_range.start for x in included)):
            raise ADRIOProcessingError(self, self.context, f"bad start {adrio_range}")
        # At least one var must have its end equal to the ADRIO range
        if not any((x.end == adrio_range.end for x in included)):
            raise ADRIOProcessingError(self, self.context, f"bad end {adrio_range}")

        source_np = self.data(PopulationByAge.POP_BY_AGE_TABLE)
        result_np = source_np[:, col_mask].sum(axis=1)

        return InspectResult(
            adrio=self,
            source=source_np,
            result=result_np,
            dtype=self.result_format.value_dtype,
            shape=self.result_format.shape,
            issues=[],
        )


class AverageHouseholdSize(_FetchACS5[np.float64]):
    """
    Retrieves an N-shaped array of floats representing the average number of people
    living in each household for every geographic node.
    Data is retrieved from Census table variable B25010_001 using ACS5 5-year estimates.
    """

    _RESULT_FORMAT = ResultFormat(
        shape=Shapes.N,
        value_dtype=np.float64,
        validation=range_mask_fn(minimum=np.float64(0), maximum=None),
        is_date_value=False,
    )

    @property
    @override
    def result_format(self) -> ResultFormat[np.float64]:
        return AverageHouseholdSize._RESULT_FORMAT

    @property
    @override
    def _variables(self) -> list[str]:
        return ["B25010_001E"]


class MedianAge(_FetchACS5[np.float64]):
    """
    Retrieves an N-shaped array of floats representing the median age in
    each geographic node. Data is retrieved from Census table variable
    B01002_001 using ACS 5-year estimates.
    """

    _RESULT_FORMAT = ResultFormat(
        shape=Shapes.N,
        value_dtype=np.float64,
        validation=range_mask_fn(minimum=np.float64(0), maximum=None),
        is_date_value=False,
    )

    @property
    @override
    def result_format(self) -> ResultFormat[np.float64]:
        return MedianAge._RESULT_FORMAT

    @property
    @override
    def _variables(self) -> list[str]:
        return ["B01002_001E"]


class MedianIncome(_FetchACS5[np.float64]):
    """
    Retrieves an N-shaped array of floats representing the median yearly income in
    each geographic node. Data is retrieved from Census table variable B19013_001
    using ACS 5-year estimates.
    """

    _RESULT_FORMAT = ResultFormat(
        shape=Shapes.N,
        value_dtype=np.float64,
        validation=range_mask_fn(minimum=np.float64(0), maximum=None),
        is_date_value=False,
    )

    @property
    @override
    def result_format(self) -> ResultFormat[np.float64]:
        return MedianIncome._RESULT_FORMAT

    @property
    @override
    def _variables(self) -> list[str]:
        return ["B19013_001E"]


class GiniIndex(_FetchACS5[np.float64]):
    """
    Retrieves an N-shaped array of floats representing the amount of income inequality
    on a scale of 0 (perfect equality) to 1 (perfect inequality) for each
    geographic node. Data is retrieved from Census table variable B19083_001 using
    ACS 5-year estimates.
    """

    # NOTE: the Gini index is named after its inventor, Corrado Gini, so this is the
    # correct capitalization

    # sentinel values: https://www.census.gov/data/developers/data-sets/acs-1year/notes-on-acs-estimate-and-annotation-values.html

    _fix_insufficient_data: Fix[np.float64]
    """
    The method to use to replace values that could not be computed due to an
    insufficient number of sample observation (-666666666 in the data).
    """
    _fix_missing: Fill[np.float64]
    """The method to use to fix missing values."""

    def __init__(
        self,
        *,
        fix_insufficient_data: Fix[np.float64]
        | float
        | Callable[[], float]
        | Literal[False] = False,
        fix_missing: Fill[np.float64]
        | float
        | Callable[[], float]
        | Literal[False] = False,
    ):
        try:
            self._fix_insufficient_data = Fix.of_float64(fix_insufficient_data)
        except ValueError:
            raise ValueError("Invalid value for `fix_insufficient_data`")
        try:
            self._fix_missing = Fill.of_float64(fix_missing)
        except ValueError:
            raise ValueError("Invalid value for `fix_missing`")

    _RESULT_FORMAT = ResultFormat(
        shape=Shapes.N,
        value_dtype=np.float64,
        validation=range_mask_fn(minimum=np.float64(0), maximum=np.float64(1.0)),
        is_date_value=False,
    )

    @property
    @override
    def result_format(self) -> ResultFormat[np.float64]:
        return GiniIndex._RESULT_FORMAT

    @property
    @override
    def _variables(self) -> list[str]:
        return ["B19083_001E"]

    @override
    def validate_context(self, context: Context) -> None:
        super().validate_context(context)
        if isinstance(context.scope, BlockGroupScope):
            err = "Gini index is not available for block group scope."
            raise ADRIOContextError(self, context, err)

    @override
    def _process(
        self,
        context: Context,
        data_df: pd.DataFrame,
    ) -> ProcessResult[np.float64]:
        vrbs = data_df["variable"].unique().tolist()
        return process_n(
            sentinels=[
                FixSentinel(
                    "insufficient_data",
                    np.float64(-666666666),
                    self._fix_insufficient_data,
                ),
            ],
            fix_missing=self._fix_missing,
            dtype=self.result_format.value_dtype,
            context=context,
            data_df=data_df,
            value_name=vrbs[0],
        )


# fmt: off
RaceCategory = Literal["White", "Black", "Native", "Asian", "Pacific Islander", "Other", "Multiple"]  # noqa: E501
"""
A racial category as defined by ACS5.

`Literal["White", "Black", "Native", "Asian", "Pacific Islander", "Other", "Multiple"]`
"""
# fmt: on


class PopulationByRace(_FetchACS5[np.int64]):
    """
    TODO

    Parameters
    ----------
    race : RaceCategory
        The Census-defined race category to load.
    """

    _RACE_VARIABLES: dict[RaceCategory, str] = {
        "White": "B02001_002E",
        "Black": "B02001_003E",
        "Native": "B02001_004E",
        "Asian": "B02001_005E",
        "Pacific Islander": "B02001_006E",
        "Other": "B02001_007E",
        "Multiple": "B02001_008E",
    }

    _RESULT_FORMAT = ResultFormat(
        shape=Shapes.N,
        value_dtype=np.int64,
        validation=range_mask_fn(minimum=np.int64(0), maximum=None),
        is_date_value=False,
    )

    _race: RaceCategory
    """The race category to load."""

    def __init__(self, race: RaceCategory):
        self._race = race

    @property
    @override
    def result_format(self) -> ResultFormat[np.int64]:
        return PopulationByRace._RESULT_FORMAT

    @property
    @override
    def _variables(self) -> list[str]:
        return [PopulationByRace._RACE_VARIABLES[self._race]]


class DissimilarityIndex(ADRIOPrototype[np.float64, np.float64]):
    """
    TODO

    Parameters
    ----------
    majority_pop : RaceCategory
        The race category representing the majority population for the amount of
        segregation.

    minority_pop : RaceCategory
        The race category representing the minority population within the
        segregation analysis.
    """

    _RESULT_FORMAT = ResultFormat(
        shape=Shapes.N,
        value_dtype=np.float64,
        validation=range_mask_fn(minimum=np.float64(0), maximum=np.float64(1.0)),
        is_date_value=False,
    )

    _majority_pop: RaceCategory
    """The race category of the majority population of interest"""
    _minority_pop: RaceCategory
    """The race category of the minority population of interest"""

    def __init__(self, majority_pop: RaceCategory, minority_pop: RaceCategory):
        self._majority_pop = majority_pop
        self._minority_pop = minority_pop

    @override
    def validate_context(self, context: Context) -> None:
        _get_scope(self, context)
        if isinstance(context.scope, BlockGroupScope):
            err = (
                "Dissimilarity index is not available for block group scope. "
                "We need to be able to load population data for the "
                "Census geography granularity one step finer than the target "
                "granularity, and block group data is already the finest available."
            )
            raise ADRIOContextError(self, context, err)

    @property
    @override
    def result_format(self) -> ResultFormat[np.float64]:
        return DissimilarityIndex._RESULT_FORMAT

    @override
    def inspect(self) -> InspectResult[np.float64, np.float64]:
        high_scope = _get_scope(self, self.context)
        high_majority = self.defer(PopulationByRace(self._majority_pop))
        high_minority = self.defer(PopulationByRace(self._minority_pop))

        low_scope = high_scope.lower_granularity()
        low_majority = self.defer(PopulationByRace(self._majority_pop), scope=low_scope)
        low_minority = self.defer(PopulationByRace(self._minority_pop), scope=low_scope)

        as_high = np.vectorize(CensusGranularity.of(high_scope.granularity).truncate)
        nodes_high = high_scope.node_ids
        nodes_low = as_high(low_scope.node_ids)

        # disaggregate high scope population to get the total for each low scope node
        high_low_index_map = np.searchsorted(nodes_high, nodes_low)
        maj_total = high_majority[high_low_index_map]
        min_total = high_minority[high_low_index_map]

        issues = []
        if np.ma.is_masked(maj_total):
            issues.append(("majority_total_unavailable", np.ma.getmask(maj_total)))
        if np.ma.is_masked(min_total):
            issues.append(("minority_total_unavailable", np.ma.getmask(min_total)))
        if np.any(maj_zeros := (maj_total == 0)):
            issues.append(("majority_total_zero", maj_zeros))
        if np.any(min_zeros := (min_total == 0)):
            issues.append(("minority_total_zero", min_zeros))

        # compute subscores where masked values are not involved
        low_mask = reduce(np.logical_or, [m for _, m in issues], np.ma.nomask)
        unmasked = ~low_mask
        subscore_np = np.zeros(shape=low_scope.nodes, dtype=np.float64)
        a = low_minority[unmasked] / min_total[unmasked]
        b = low_majority[unmasked] / maj_total[unmasked]
        subscore_np[unmasked] = np.abs(a - b)

        # aggregate subscores back to high scope
        result_np = 0.5 * np.bincount(high_low_index_map, weights=subscore_np)

        def aggregate_mask(mask):
            return np.bincount(high_low_index_map, weights=mask).astype(np.bool_)

        if np.any(low_mask):
            issues = [(issue_name, aggregate_mask(m)) for issue_name, m in issues]
            result_np = np.ma.masked_array(result_np, aggregate_mask(low_mask))

        return InspectResult(
            adrio=self,
            source=np.column_stack((low_majority, low_minority, maj_total, min_total)),
            result=result_np,
            dtype=self.result_format.value_dtype,
            shape=self.result_format.shape,
            issues=issues,
        )
