import os
from abc import abstractmethod
from collections import defaultdict
from functools import cache
from json import load as load_json
from typing import Callable, Literal, Sequence

import numpy as np
import pandas as pd
import requests
from numpy.typing import NDArray
from typing_extensions import override

from epymorph.adrio.adrio import (
    ADRIOCommunicationError,
    ADRIOContextError,
    ADRIOError,
    ADRIOPrototype,
    DontFill,
    Fill,
    Fix,
    FixSentinel,
    ProcessResult,
    ResultFormat,
    ValueT,
    process_n,
    range_mask_fn,
)
from epymorph.cache import load_or_fetch_url, module_cache_path
from epymorph.data_shape import Shapes
from epymorph.geography.us_census import (
    BlockGroupScope,
    CensusScope,
    CountyScope,
    StateScope,
    TractScope,
)
from epymorph.geography.us_geography import BLOCK_GROUP, COUNTY, STATE, TRACT
from epymorph.simulation import Context


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


class _ACS5(ADRIOPrototype[ValueT, ValueT]):
    @property
    @abstractmethod
    def _variables(self) -> list[str]:
        """The ACS variables to fetch for this ADRIO."""

    def _get_scope(self, context: Context) -> CensusScope:
        if not isinstance(context.scope, CensusScope):
            err = "US Census geo scope required."
            raise ADRIOContextError(self, context, err)
        if context.scope.year not in ACS5_YEARS:
            err = f"{context.scope.year} is not a supported year for ACS5 data."
            raise ADRIOContextError(self, context, err)
        return context.scope

    @override
    def validate_context(self, context: Context):
        self._get_scope(context)
        if census_api_key() is None:
            err = (
                "Census API key is required for accessing ACS5 data. "
                "Please set the environment variable 'CENSUS_API_KEY'"
            )
            raise ADRIOContextError(self, context, err)

    @override
    def _fetch(self, context: Context) -> pd.DataFrame:
        scope = self._get_scope(context)
        vrbs = self._variables
        vrb_slice = np.s_[1 : len(vrbs) + 1]

        url = ACS5Client.url(scope.year)
        key = census_api_key()
        get = ",".join(["GEO_ID", *vrbs])
        value_dtype = self.result_format.value_dtype

        queries = ACS5Client.make_queries(scope)
        processing_steps = len(queries) + 1
        with requests.Session() as session:
            try:
                results = []
                for i, q in enumerate(queries):
                    response = session.get(
                        url,
                        params={"key": key, "get": get, **q},
                        timeout=30,
                    )
                    response.raise_for_status()  # Raise an error for bad status codes
                    [columns, *rows] = response.json()

                    # convert rows to numpy array, then to DataFrame
                    result_np = np.array(rows, dtype=np.str_)
                    result_df = pd.DataFrame(
                        # drop geoid prefix "0500000US"
                        index=pd.Index(result_np[:, 0], name="geoid").str.slice(9),
                        data=result_np[:, vrb_slice].astype(value_dtype),
                        columns=columns[vrb_slice],
                    )

                    results.append(result_df)
                    self.report_progress((i + 1) / processing_steps)

                return pd.concat(results).sort_index()
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

    @override
    def _validate_result(
        self,
        context: Context,
        result: NDArray[ValueT],
        *,
        expected_shape: tuple[int, ...] | None = None,
    ) -> None:
        num_vars = len(self._variables)
        shp = None if num_vars == 1 else (context.scope.nodes, num_vars)
        super()._validate_result(context, result, expected_shape=shp)


class Population(_ACS5[np.int64]):
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


class PopulationByAgeTable(_ACS5[np.int64]):
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
        scope = self._get_scope(self.context)
        return ACS5Client.get_group_var_names(scope.year, "B01001")


class AverageHouseholdSize(_ACS5[np.float64]):
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


class MedianAge(_ACS5[np.float64]):
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


class MedianIncome(_ACS5[np.float64]):
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


class GiniIndex(_ACS5[np.float64]):
    """
    Retrieves an N-shaped array of floats representing the amount of income inequality
    on a scale of 0 (perfect equality) to 1 (perfect inequality) for each
    geographic node. Data is retrieved from Census table variable B19083_001 using
    ACS 5-year estimates.
    """

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
        validation=range_mask_fn(minimum=np.float64(0), maximum=None),
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
        )


# TODO:
# PopulationByAge
# DissimilarityIndex
