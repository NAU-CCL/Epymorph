import os
import re
from abc import abstractmethod
from functools import cache, reduce
from itertools import groupby
from json import load as load_json
from typing import Callable, Iterable, Literal, NamedTuple, Sequence, TypeGuard

import numpy as np
import pandas as pd
import requests
from numpy.typing import NDArray
from typing_extensions import override

from epymorph.adrio.adrio import (
    ADRIO,
    ADRIOCommunicationError,
    ADRIOContextError,
    ADRIOProcessingError,
    FetchADRIO,
    InspectResult,
    ProcessResult,
    ResultFormat,
    ValueT,
    adrio_cache,
    range_mask_fn,
)
from epymorph.adrio.processing import (
    DataPipeline,
    DontFill,
    DontFix,
    Fill,
    Fix,
    PivotAxis,
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
    TRACT,
    CensusGranularity,
)
from epymorph.simulation import Context
from epymorph.util import filter_with_mask


def census_api_key() -> str | None:
    """
    Loads the API key to use for census.gov,
    as environment variable 'API_KEY__census.gov'.
    If that's not found we fall back to 'CENSUS_API_KEY',
    as a legacy form.
    """
    key = os.environ.get("API_KEY__census.gov", default=None)
    if key is None:
        key = os.environ.get("CENSUS_API_KEY", default=None)
    return key


# fmt:off
ACS5Year = Literal[2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]  # noqa: E501
"""A supported ACS5 data year."""

ACS5_YEARS: Sequence[ACS5Year] = (2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023)  # noqa: E501
"""All supported ACS5 data years."""
# fmt: on

_ACS5_CACHE_PATH = module_cache_path(__name__)
"""
For caching ACS5. At the moment, the only thing that is cached is variables metadata.
"""


class ACS5Client:
    """Methods for interacting with the Census API for ACS5 data."""

    @staticmethod
    def url(year: int) -> str:
        """The base request URL for a given ACS5 year."""
        return f"https://api.census.gov/data/{year}/acs/acs5"

    @cache
    @staticmethod
    def get_vars(year: int) -> dict[str, dict]:
        """Loads (and caches) ACS5 variable metadata."""
        try:
            vars_url = f"{ACS5Client.url(year)}/variables.json"
            cache_path = _ACS5_CACHE_PATH / f"variables-{year}.json"
            file = load_or_fetch_url(vars_url, cache_path)
            return load_json(file)["variables"]
        except Exception as e:
            err = "Unable to load ACS5 variables."
            raise Exception(err) from e

    @cache
    @staticmethod
    def get_group_vars(year: int, group: str) -> list[tuple[str, dict]]:
        """Retrieves the variables metadata for a specific group of variables."""
        variables = sorted(
            (
                (name, attrs)
                for name, attrs in ACS5Client.get_vars(year).items()
                if attrs["group"] == group
            ),
            key=lambda x: x[0],
        )
        if len(variables) == 0:
            raise ValueError(f"ACS5 variable group '{group}' not found in year {year}.")
        return variables

    @cache
    @staticmethod
    def get_group_var_names(year: int, group: str) -> list[str]:
        """Like `get_group_vars` but just returns the variable names in the group."""
        return [var for var, _ in ACS5Client.get_group_vars(year, group)]

    @staticmethod
    def make_queries(scope: CensusScope) -> list[dict[str, str]]:
        """
        Creates one or more Census API query predicates for the given scope.
        These may involve the "for" and "in" request parameters.
        Depending on your scope and the limitations of the API, multiple queries
        may be required, especially when your scope represents a disjoint spatial
        selection or one that otherwise can't be neatly expressed in a form like
        "all counties within state X".
        """

        def to_list(group: Iterable[tuple[str, ...]]) -> str:
            return ",".join(map(lambda x: x[-1], group))

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
                return [
                    {"for": f"county:{to_list(group)}", "in": f"state:{state}"}
                    for (state,), group in groupby(
                        map(COUNTY.decompose, includes),
                        key=lambda x: x[0:-1],
                    )
                ]

            case TractScope(includes_granularity="state", includes=includes):
                return [
                    {
                        "for": "tract:*",
                        "in": f"state:{','.join(includes)} county:*",
                    }
                ]

            case TractScope(includes_granularity="county", includes=includes):
                return [
                    {"for": "tract:*", "in": f"state:{state} county:{to_list(group)}"}
                    for (state,), group in groupby(
                        map(COUNTY.decompose, includes),
                        key=lambda x: x[0:-1],
                    )
                ]

            case TractScope(includes_granularity="tract", includes=includes):
                return [
                    {
                        "for": f"tract:{to_list(group)}",
                        "in": f"state:{state} county:{county}",
                    }
                    for (state, county), group in groupby(
                        map(TRACT.decompose, includes),
                        key=lambda x: x[0:-1],
                    )
                ]

            case BlockGroupScope(includes_granularity="state", includes=includes):
                # This wouldn't normally need to be multiple queries,
                # but Census API won't let you fetch CBGs for multiple states.
                return [
                    {"for": "block group:*", "in": f"state:{state} county:* tract:*"}
                    for state in includes
                ]

            case BlockGroupScope(includes_granularity="county", includes=includes):
                return [
                    {
                        "for": "block group:*",
                        "in": f"state:{state} county:{to_list(group)} tract:*",
                    }
                    for (state,), group in groupby(
                        map(COUNTY.decompose, includes),
                        key=lambda x: x[0:-1],
                    )
                ]

            case BlockGroupScope(includes_granularity="tract", includes=includes):
                return [
                    {
                        "for": "block group:*",
                        "in": f"state:{state} county:{county} tract:{to_list(group)}",
                    }
                    for (state, county), group in groupby(
                        map(TRACT.decompose, includes),
                        key=lambda x: x[0:-1],
                    )
                ]

            case BlockGroupScope(includes_granularity="block group", includes=includes):
                return [
                    {
                        "for": f"block group:{to_list(group)}",
                        "in": f"state:{state} county:{county} tract:{tract}",
                    }
                    for (state, county, tract), group in groupby(
                        map(BLOCK_GROUP.decompose, includes),
                        key=lambda x: x[0:-1],
                    )
                ]

            case _:
                err = "Unsupported geo scope."
                raise Exception(err)

    @staticmethod
    def fetch(
        scope: CensusScope,
        variables: list[str],
        value_dtype: type[np.generic],
        report_progress: Callable[[float], None] | None = None,
    ) -> pd.DataFrame:
        """
        Requests `variables` from the Census API for the given `scope`.
        Returns a DataFrame in "long" format, with columns: geoid, variable, and value.
        Geoid and variable are strings and value will be converted to the given dtype.
        """
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
                if report_progress:
                    report_progress((i + 1) / processing_steps)

            return pd.concat(results).sort_index()


def _get_scope(adrio: ADRIO, context: Context) -> CensusScope:
    """
    Validate and retrieve the CensusScope from `context`.
    ACS5 ADRIOs really only work on Census geography, so this is a commonly-useful
    validation function.
    """
    if not isinstance(context.scope, CensusScope):
        err = "US Census geo scope required."
        raise ADRIOContextError(adrio, context, err)
    if context.scope.year not in ACS5_YEARS:
        err = f"{context.scope.year} is not a supported year for ACS5 data."
        raise ADRIOContextError(adrio, context, err)
    return context.scope


class _FetchACS5(FetchADRIO[ValueT, ValueT]):
    """
    An abstract specialization of `FetchADRIO` for ADRIOs which fetch
    data from the ACS5 API. At a minimum, implementors will need to
    override `result_format` and `_variables`, and that is sufficient for
    simple cases.
    """

    # sentinel values: https://www.census.gov/data/developers/data-sets/acs-1year/notes-on-acs-estimate-and-annotation-values.html

    _fix_insufficient_data: Fix[ValueT]
    """
    The method to use to replace values that could not be computed due to an
    insufficient number of sample observation (-666666666 in the data).
    """
    _fix_missing: Fill[ValueT]
    """The method to use to fix missing values."""

    def __init__(
        self,
        *,
        fix_insufficient_data: Fix[ValueT] | None = None,
        fix_missing: Fill[ValueT] | None = None,
    ):
        self._fix_insufficient_data = fix_insufficient_data or DontFix()
        self._fix_missing = fix_missing or DontFill()

    @property
    @abstractmethod
    def _variables(self) -> list[str]:
        """The ACS variables to fetch for this ADRIO."""

    _census_scope: CensusScope | None = None

    @override
    def validate_context(self, context: Context):
        self._census_scope = _get_scope(self, context)
        if census_api_key() is None:
            err = (
                "Census API key is required for accessing ACS5 data. "
                "Please set the environment variable 'CENSUS_API_KEY'"
            )
            raise ADRIOContextError(self, context, err)

    @property
    def census_scope(self) -> CensusScope:
        """
        Gets the context scope type-narrowed to CensusScope.
        If for any reason this has not yet been verified, it first validates this fact.
        """
        if self._census_scope is None:
            self._census_scope = _get_scope(self, self.context)
        return self._census_scope

    @override
    def _fetch(self, context: Context) -> pd.DataFrame:
        try:
            return ACS5Client.fetch(
                scope=self.census_scope,
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

        # If we're loading a group variable, expand it.
        if len(vrbs) == 1 and (m := re.match(r"^group\((.*?)\)$", vrbs[0])):
            group_name = m.group(1)
            vrbs = ACS5Client.get_group_var_names(self.census_scope.year, group_name)

        pipeline = (
            DataPipeline(
                axes=(
                    PivotAxis("geoid", context.scope.node_ids),
                    PivotAxis("variable", vrbs),
                ),
                ndims=1 if len(vrbs) == 1 else 2,
                dtype=self.result_format.value_dtype,
                rng=context,
            )
            .strip_sentinel(
                "insufficient_data",
                np.float64(-666666666),
                self._fix_insufficient_data,
            )
            .finalize(self._fix_missing)
        )
        return pipeline(data_df)


@adrio_cache
class Population(_FetchACS5[np.int64]):
    """
    Loads population data from the US Census ACS 5-Year Data (variable B01001_001E).
    ACS5 data is compiled from surveys taken during a rolling five year period, and
    as such are estimates.

    Data is available using CensusScope geographies, from StateScope down to
    BlockGroupScope (aggregates are computed by the Census Bureau). Data is loaded
    according to the scope's year, from 2009 to 2023.

    The result is an N-shaped array of integers.

    See Also
    --------
    The [ACS 5-Year documentation](https://www.census.gov/data/developers/data-sets/acs-5year.html)
    from the US Census.
    """

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


@adrio_cache
class PopulationByAgeTable(_FetchACS5[np.int64]):
    """
    Loads a table of population categorized by Census-defined age brackets from the
    US Census ACS 5-Year Data (group B01001). This table is most useful as the source
    data for one or more [`PopulationByAge`](`epymorph.adrio.acs5.PopulationByAge`)
    ADRIOs, which knows how to select, group, and aggregate the data for simulations.
    ACS5 data is compiled from surveys taken during a rolling five year period, and
    as such are estimates.

    Data is available using CensusScope geographies, from StateScope down to
    BlockGroupScope (aggregates are computed by the Census Bureau). Data is loaded
    according to the scope's year, from 2009 to 2023.

    The result is an NxA-shaped array of integers where A is the number of variables
    included in the table. For example, in 2023 there are 49 variables: 23 age brackets
    for male, 23 age brackets for female, the male all-ages total, the female all-ages
    total, and a grand total.

    See Also
    --------
    The [ACS 5-Year documentation](https://www.census.gov/data/developers/data-sets/acs-5year.html)
    from the US Census, and [an example of this table for 2023](https://data.census.gov/table/ACSDT5Y2023.B01001).
    """

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
        vrbs = ACS5Client.get_group_var_names(self.census_scope.year, "B01001")
        shp = (context.scope.nodes, len(vrbs))
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

    AgeRange is a NamedTuple.
    """

    start: int
    """The youngest age included in the range."""
    end: int | None
    """The oldest age included in the range, or None to indicate an unbounded range."""

    def contains(self, other: "AgeRange") -> bool:
        """
        Is the `other` range fully contained in (or coincident with) this range?

        Parameters
        ----------
        other : AgeRange
            The other age range to consider.

        Returns
        -------
        bool
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
        Parse the age range of an ACS field label;
        e.g.: `Estimate!!Total:!!Male:!!22 to 24 years`

        Parameters
        ----------
        label : str
            A census variable label.

        Returns
        -------
        AgeRange | None
            The AgeRange object if parsing is successful, None if not.
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


@adrio_cache
class PopulationByAge(ADRIO[np.int64, np.int64]):
    """
    Processes a population-by-age table to extract the population of a specified age
    bracket, as limited by the age brackets defined by the US Census ACS 5-Year Data
    (group B01001). This ADRIO does not fetch data on its own, but requires you to
    provide another attribute named "population_by_age_table" for it to parse.
    Most often, this will be provided by a
    [`PopulationByAgeTable`](`epymorph.adrio.acs5.PopulationByAgeTable`) instance.
    This allows the table to be reused in case you need to calculate more than one
    population bracket (as is common in a multi-strata model).

    The result is an N-shaped array of integers.

    Parameters
    ----------
    age_range_start : int
        The youngest age to include in the age bracket.
    age_range_end : int | None
        The oldest age to include in the age bracket, or None to indicate an unbounded
        range (include all ages greater than or equal to `age_range_start`).

    Raises
    ------
    ValueError
        If the given age range does not line up with those ranges which are available
        in the source data. For instance, the Census defines an age bracket of 20-to-24
        years. This makes it impossible for 21, 22, or 23 to be either the start or end
        of an age range. You can view the available age ranges on
        [data.census.gov](https://data.census.gov/table/ACSDT5Y2023.B01001).

    See Also
    --------
    The [ACS 5-Year documentation](https://www.census.gov/data/developers/data-sets/acs-5year.html)
    from the US Census, and [an example of this table for 2023](https://data.census.gov/table/ACSDT5Y2023.B01001).
    """

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
    """Defines the population-by-age-table requirement of this ADRIO."""

    requirements = (POP_BY_AGE_TABLE,)

    _age_range: AgeRange
    """The age range to load with this ADRIO."""

    def __init__(self, age_range_start: int, age_range_end: int | None):
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
        self.validate_context(self.context)
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
            issues={},
        )


# fmt: off
RaceCategory = Literal["White", "Black", "Native", "Asian", "Pacific Islander", "Other", "Multiple"]  # noqa: E501
"""
A racial category defined by ACS5.

`Literal["White", "Black", "Native", "Asian", "Pacific Islander", "Other", "Multiple"]`
"""
# fmt: on


@adrio_cache
class PopulationByRace(_FetchACS5[np.int64]):
    """
    Loads population by race from the US Census ACS 5-Year Data (group B02001).
    ACS5 data is compiled from surveys taken during a rolling five year period, and
    as such are estimates.

    Data is available using CensusScope geographies, from StateScope down to
    BlockGroupScope (aggregates are computed by the Census Bureau). Data is loaded
    according to the scope's year, from 2009 to 2023.

    The result is an N-shaped array of integers.

    Parameters
    ----------
    race : RaceCategory
        The Census-defined race category to load.
    fix_insufficient_data : Fill[np.int64] | int | Callable[[], int] | Literal[False], default=False
        The method to use to fix values for which there were insufficient data to report
        (sentinel value: -666666666).
    fix_missing : Fill[np.int64] | int | Callable[[], int] | Literal[False], default=False
        The method to use to fix missing values.

    See Also
    --------
    The [ACS 5-Year documentation](https://www.census.gov/data/developers/data-sets/acs-5year.html)
    from the US Census.
    """  # noqa: E501

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

    def __init__(
        self,
        race: RaceCategory,
        *,
        fix_insufficient_data: Fix[np.int64]
        | int
        | Callable[[], int]
        | Literal[False] = False,
        fix_missing: Fill[np.int64] | int | Callable[[], int] | Literal[False] = False,
    ):
        self._race = race
        try:
            fix_insufficient_data = Fix.of_int64(fix_insufficient_data)
        except ValueError:
            raise ValueError("Invalid value for `fix_insufficient_data`")
        try:
            fix_missing = Fill.of_int64(fix_missing)
        except ValueError:
            raise ValueError("Invalid value for `fix_missing`")
        super().__init__(
            fix_insufficient_data=fix_insufficient_data,
            fix_missing=fix_missing,
        )

    @property
    @override
    def result_format(self) -> ResultFormat[np.int64]:
        return PopulationByRace._RESULT_FORMAT

    @property
    @override
    def _variables(self) -> list[str]:
        return [PopulationByRace._RACE_VARIABLES[self._race]]


@adrio_cache
class AverageHouseholdSize(_FetchACS5[np.float64]):
    """
    Loads average household size data, based on the number of people living in a
    household, from the US Census ACS 5-Year Data (variable B25010_001E).
    ACS5 data is compiled from surveys taken during a rolling five year period, and
    as such are estimates.

    Data is available using CensusScope geographies, from StateScope down to
    BlockGroupScope (aggregates are computed by the Census Bureau). Data is loaded
    according to the scope's year, from 2009 to 2023.

    The result is an N-shaped array of floats.

    Parameters
    ----------
    fix_insufficient_data : Fill[np.float64] | int | Callable[[], int] | Literal[False], default=False
        The method to use to fix values for which there were insufficient data to report
        (sentinel value: -666666666).
    fix_missing : Fill[np.float64] | int | Callable[[], int] | Literal[False], default=False
        The method to use to fix missing values.

    See Also
    --------
    The [ACS 5-Year documentation](https://www.census.gov/data/developers/data-sets/acs-5year.html)
    from the US Census.
    """  # noqa: E501

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
            fix_insufficient_data = Fix.of_float64(fix_insufficient_data)
        except ValueError:
            raise ValueError("Invalid value for `fix_insufficient_data`")
        try:
            fix_missing = Fill.of_float64(fix_missing)
        except ValueError:
            raise ValueError("Invalid value for `fix_missing`")
        super().__init__(
            fix_insufficient_data=fix_insufficient_data,
            fix_missing=fix_missing,
        )

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


@adrio_cache
class MedianAge(_FetchACS5[np.float64]):
    """
    Loads median age data from the US Census ACS 5-Year Data (variable B01002_001E).
    ACS5 data is compiled from surveys taken during a rolling five year period, and
    as such are estimates.

    Data is available using CensusScope geographies, from StateScope down to
    BlockGroupScope (aggregates are computed by the Census Bureau). Data is loaded
    according to the scope's year, from 2009 to 2023.

    The result is an N-shaped array of floats.

    Parameters
    ----------
    fix_insufficient_data : Fill[np.float64] | int | Callable[[], int] | Literal[False], default=False
        The method to use to fix values for which there were insufficient data to report
        (sentinel value: -666666666).
    fix_missing : Fill[np.float64] | int | Callable[[], int] | Literal[False], default=False
        The method to use to fix missing values.

    See Also
    --------
    The [ACS 5-Year documentation](https://www.census.gov/data/developers/data-sets/acs-5year.html)
    from the US Census.
    """  # noqa: E501

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
            fix_insufficient_data = Fix.of_float64(fix_insufficient_data)
        except ValueError:
            raise ValueError("Invalid value for `fix_insufficient_data`")
        try:
            fix_missing = Fill.of_float64(fix_missing)
        except ValueError:
            raise ValueError("Invalid value for `fix_missing`")
        super().__init__(
            fix_insufficient_data=fix_insufficient_data,
            fix_missing=fix_missing,
        )

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


@adrio_cache
class MedianIncome(_FetchACS5[np.int64]):
    """
    Loads median income data in whole dollars from the US Census ACS 5-Year Data
    (variable B19013_001E), which is adjusted for inflation to the year of the data.
    ACS5 data is compiled from surveys taken during a rolling five year period, and
    as such are estimates.

    Data is available using CensusScope geographies, from StateScope down to
    BlockGroupScope (aggregates are computed by the Census Bureau). Data is loaded
    according to the scope's year, from 2009 to 2023.

    The result is an N-shaped array of integers.

    Parameters
    ----------
    fix_insufficient_data : Fill[np.int64] | int | Callable[[], int] | Literal[False], default=False
        The method to use to fix values for which there were insufficient data to report
        (sentinel value: -666666666).
    fix_missing : Fill[np.int64] | int | Callable[[], int] | Literal[False], default=False
        The method to use to fix missing values.

    See Also
    --------
    The [ACS 5-Year documentation](https://www.census.gov/data/developers/data-sets/acs-5year.html)
    from the US Census.
    """  # noqa: E501

    def __init__(
        self,
        *,
        fix_insufficient_data: Fix[np.int64]
        | int
        | Callable[[], int]
        | Literal[False] = False,
        fix_missing: Fill[np.int64] | int | Callable[[], int] | Literal[False] = False,
    ):
        try:
            fix_insufficient_data = Fix.of_int64(fix_insufficient_data)
        except ValueError:
            raise ValueError("Invalid value for `fix_insufficient_data`")
        try:
            fix_missing = Fill.of_int64(fix_missing)
        except ValueError:
            raise ValueError("Invalid value for `fix_missing`")
        super().__init__(
            fix_insufficient_data=fix_insufficient_data,
            fix_missing=fix_missing,
        )

    _RESULT_FORMAT = ResultFormat(
        shape=Shapes.N,
        value_dtype=np.int64,
        validation=range_mask_fn(minimum=np.int64(0), maximum=None),
        is_date_value=False,
    )

    @property
    @override
    def result_format(self) -> ResultFormat[np.int64]:
        return MedianIncome._RESULT_FORMAT

    @property
    @override
    def _variables(self) -> list[str]:
        return ["B19013_001E"]


@adrio_cache
class GiniIndex(_FetchACS5[np.float64]):
    """
    Loads Gini Index data from the US Census ACS 5-Year Data (variable B19083_001E).
    This is a measure of income inequality on a scale from 0 (perfect equality)
    to 1 (perfect inequality).
    ACS5 data is compiled from surveys taken during a rolling five year period, and
    as such are estimates.

    Data is available using CensusScope geographies, from StateScope down to
    BlockGroupScope (aggregates are computed by the Census Bureau). Data is loaded
    according to the scope's year, from 2009 to 2023.

    The result is an N-shaped array of floats.

    Parameters
    ----------
    fix_insufficient_data : Fill[np.float64] | int | Callable[[], int] | Literal[False], default=False
        The method to use to fix values for which there were insufficient data to report
        (sentinel value: -666666666).
    fix_missing : Fill[np.float64] | int | Callable[[], int] | Literal[False], default=False
        The method to use to fix missing values.

    See Also
    --------
    The [ACS 5-Year documentation](https://www.census.gov/data/developers/data-sets/acs-5year.html)
    from the US Census, and [general info on the Gini index](https://en.wikipedia.org/wiki/Gini_coefficient).
    """  # noqa: E501

    # NOTE: the Gini index is named after its inventor, Corrado Gini, so this is the
    # correct capitalization

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
            fix_insufficient_data = Fix.of_float64(fix_insufficient_data)
        except ValueError:
            raise ValueError("Invalid value for `fix_insufficient_data`")
        try:
            fix_missing = Fill.of_float64(fix_missing)
        except ValueError:
            raise ValueError("Invalid value for `fix_missing`")
        super().__init__(
            fix_insufficient_data=fix_insufficient_data,
            fix_missing=fix_missing,
        )

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


@adrio_cache
class DissimilarityIndex(ADRIO[np.float64, np.float64]):
    """
    Calculates the Dissimilarity Index using US Census ACS 5-Year Data (group B02001).
    The dissimilarity index is a measure of segregation comparing two races.
    Typically one compares a majority to a minority race and so the names of parameters
    reflect this, but this relationship between races involved isn't strictly necessary.
    The numerical result can be interpreted as the percentage of "minority" individuals
    that would have to move in order for the geographic distribution of individuals
    within subdivisions of a location to match the distribution of individuals in the
    location as a whole.
    ACS5 data is compiled from surveys taken during a rolling five year period, and
    as such are estimates.

    Data is available using CensusScope geographies, from StateScope down to
    TractScope. Data is loaded according to the scope's year, from 2009 to 2023.
    This ADRIO does not support BlockGroupScope because we the calculation of the index
    requires loading data at a finer granularity than the target granularity, and
    there is no ACS5 data below block groups.

    The result is an N-shaped array of floats.

    Parameters
    ----------
    majority_pop : RaceCategory
        The race category representing the majority population for the amount of
        segregation.
    minority_pop : RaceCategory
        The race category representing the minority population within the
        segregation analysis.
    fix_insufficient_population : Fill[np.int64] | int | Callable[[], int] | Literal[False], default=False
        The method to use to fix values for which there were insufficient data to report
        (sentinel value: -666666666).
        The replacement is performed on the underlying population by race data.
    fix_missing_population : Fill[np.int64] | int | Callable[[], int] | Literal[False], default=False
        The method to use to fix missing values. The replacement is performed on the
        underlying population by race data.
    fix_not_computable : Fill[np.float64] | float | Callable[[], float] | Literal[False], default=False
        The method to use to fix values for which we cannot compute a value because population
        numbers cannot be loaded for one or more of the populations involved.


    See Also
    --------
    The [ACS 5-Year documentation](https://www.census.gov/data/developers/data-sets/acs-5year.html)
    from the US Census, and
    [general information about the dissimilarity index](https://en.wikipedia.org/wiki/Index_of_dissimilarity).
    """  # noqa: E501

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
    _fix_insufficient_population: Fix[np.int64]
    """
    The method to use to replace population values that could not be computed due to an
    insufficient number of sample observation (-666666666 in the data).
    """
    _fix_missing_population: Fill[np.int64]
    """The method to use to fix missing population values."""
    _fix_minority_total_zero: Fix[np.float64]
    """
    The method to use to replace dissimilarity index values when the minority population
    is zero (which would cause a divide-by-zero error).
    """

    def __init__(
        self,
        majority_pop: RaceCategory,
        minority_pop: RaceCategory,
        *,
        fix_insufficient_population: Fix[np.int64]
        | int
        | Callable[[], int]
        | Literal[False] = False,
        fix_missing_population: Fill[np.int64]
        | int
        | Callable[[], int]
        | Literal[False] = False,
        fix_not_computable: Fix[np.float64]
        | float
        | Callable[[], float]
        | Literal[False] = False,
    ):
        self._majority_pop = majority_pop
        self._minority_pop = minority_pop
        try:
            self._fix_insufficient_population = Fix.of_int64(
                fix_insufficient_population
            )
        except ValueError:
            raise ValueError("Invalid value for `fix_insufficient_population`")
        try:
            self._fix_missing_population = Fill.of_int64(fix_missing_population)
        except ValueError:
            raise ValueError("Invalid value for `fix_missing_population`")
        try:
            self._fix_not_computable = Fix.of_float64(fix_not_computable)
        except ValueError:
            raise ValueError("Invalid value for `fix_not_computable`")

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
        self.validate_context(self.context)

        majority_race = PopulationByRace(
            race=self._majority_pop,
            fix_insufficient_data=self._fix_insufficient_population,
            fix_missing=self._fix_missing_population,
        )
        minority_race = PopulationByRace(
            race=self._minority_pop,
            fix_insufficient_data=self._fix_insufficient_population,
            fix_missing=self._fix_missing_population,
        )

        # Load populations at the coarse-granularity scope (coarse nodes)
        high_scope = _get_scope(self, self.context)
        high_majority = self.defer(majority_race)
        high_minority = self.defer(minority_race)

        # Load populations at the fine-granularity scope (fine nodes)
        low_scope = high_scope.lower_granularity()
        low_majority = self.defer(majority_race, scope=low_scope)
        low_minority = self.defer(minority_race, scope=low_scope)

        # Which fine nodes belong to which coarse nodes?
        as_high = np.vectorize(CensusGranularity.of(high_scope.granularity).truncate)
        nodes_high = high_scope.node_ids
        nodes_low = as_high(low_scope.node_ids)

        # disaggregate coarse node population to get the total to use for each fine node
        high_low_index_map = np.searchsorted(nodes_high, nodes_low)
        maj_total = high_majority[high_low_index_map]
        min_total = high_minority[high_low_index_map]

        # If coarse population is unavailable or zero, we can't compute DI
        issues: list[tuple[str, NDArray[np.bool_]]] = []
        if np.ma.is_masked(maj_total):
            issues.append(("majority_total_unavailable", np.ma.getmask(maj_total)))
        if np.ma.is_masked(min_total):
            issues.append(("minority_total_unavailable", np.ma.getmask(min_total)))
        if np.any(maj_zeros := (maj_total == 0)):
            issues.append(("majority_total_zero", maj_zeros))
        if np.any(min_zeros := (min_total == 0)):
            issues.append(("minority_total_zero", min_zeros))

        # Compute scores for fine nodes (excluding where masked values are involved)
        low_mask = reduce(np.logical_or, [m for _, m in issues], np.ma.nomask)
        unmasked = ~low_mask
        subscore_np = np.zeros(shape=low_scope.nodes, dtype=np.float64)
        a = low_minority[unmasked] / min_total[unmasked]
        b = low_majority[unmasked] / maj_total[unmasked]
        subscore_np[unmasked] = np.abs(a - b)

        # Aggregate fine scores to coarse scores
        result_np = 0.5 * np.bincount(high_low_index_map, weights=subscore_np)

        # Aggregate fine masks to coarse masks
        def aggregate_mask(mask):
            return np.bincount(high_low_index_map, weights=mask).astype(np.bool_)

        if np.any(low_mask):
            issues = [(issue_name, aggregate_mask(m)) for issue_name, m in issues]
            result_np = np.ma.masked_array(result_np, aggregate_mask(low_mask))

        # If there are incomputable values and a fix, apply it;
        # if successful, consider all underlying issues addressed.
        has_fix = not isinstance(self._fix_not_computable, DontFix)
        if np.ma.is_masked(result_np) and has_fix:
            sentinel = np.float64(-999999)
            sentinel_result = np.ma.getdata(result_np).copy()
            sentinel_result[np.ma.getmask(result_np)] = sentinel
            fixed_result = self._fix_not_computable(
                rng=self.context,
                replace=sentinel,
                data_df=pd.DataFrame({"value": sentinel_result}),
            )["value"].to_numpy()
            unfixed = fixed_result == sentinel
            if unfixed.any():
                issues = [("not_computable", unfixed)]
                result_np = np.ma.masked_array(fixed_result, unfixed)
            else:
                issues = []
                result_np = fixed_result

        return InspectResult(
            adrio=self,
            source=np.column_stack((low_majority, low_minority, maj_total, min_total)),
            result=result_np,
            dtype=self.result_format.value_dtype,
            shape=self.result_format.shape,
            issues={k: v for k, v in issues},
        )
