"""ADRIOs for US Census Bureau American Community Survey Commuting Flows data."""

from pathlib import Path
from typing import Callable, Literal, NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing_extensions import override

from epymorph.adrio.adrio import (
    ADRIOCommunicationError,
    ADRIOContextError,
    FetchADRIO,
    PipelineResult,
    adrio_validate_pipe,
)
from epymorph.adrio.processing import DataPipeline, Fill, PivotAxis
from epymorph.adrio.validation import (
    ResultFormat,
    validate_dtype,
    validate_numpy,
    validate_shape,
    validate_values_in_range,
)
from epymorph.cache import check_file_in_cache, load_or_fetch_url, module_cache_path
from epymorph.data_shape import Shapes
from epymorph.data_usage import AvailableDataEstimate, DataEstimate
from epymorph.geography.us_census import CountyScope, StateScope
from epymorph.simulation import Context

_COMMFLOWS_CACHE_PATH = module_cache_path(__name__)


class _Config(NamedTuple):
    """Configuration for an ACS Comm Flows product."""

    year: int
    """The nominal year of the survey results."""
    geo_year: int
    """The Census vintage used in the data."""
    url: str
    """The URL for the source file."""
    header: int
    """How many header rows?"""
    footer: int
    """How many footer rows?"""
    cols: list[str]
    """Column names."""
    estimate: int
    """Estimated file size in bytes."""

    @property
    def cache_path(self) -> Path:
        """The path to where the source data should be cached."""
        return _COMMFLOWS_CACHE_PATH / f"{self.year}.xlsx"

    @property
    def cache_key(self) -> str:
        """The cache key to use for this result."""
        return f"commflows:{self.year}"


# fmt: off
_CONFIG = [
    _Config(
        year=2010,
        geo_year=2010,
        url="https://www2.census.gov/programs-surveys/demo/tables/metro-micro/2010/commuting-employment-2010/table1.xlsx",
        header=4,
        footer=3,
        cols=[
            "res_state_code", "res_county_code", "wrk_state_code", "wrk_county_code",
            "workers", "moe", "res_state", "res_county", "wrk_state", "wrk_county",
        ],
        estimate=7_200_000,
    ),
    _Config(
        year=2015,
        geo_year=2015,
        url="https://www2.census.gov/programs-surveys/demo/tables/metro-micro/2015/commuting-flows-2015/table1.xlsx",
        header=6,
        footer=2,
        cols=[
            "res_state_code", "res_county_code", "res_state", "res_county",
            "wrk_state_code", "wrk_county_code", "wrk_state", "wrk_county",
            "workers", "moe",
        ],
        estimate=6_700_000,
    ),
    _Config(
        year=2020,
        geo_year=2022,
        # yes, the 2020 results use 2022 geography, which is when the Census officially
        # switched to using planning regions instead of counties for Connecticut.
        # The footer of this document says as much.
        url="https://www2.census.gov/programs-surveys/demo/tables/metro-micro/2020/commuting-flows-2020/table1.xlsx",
        header=7,
        footer=4,
        cols=[
            "res_state_code", "res_county_code", "res_state", "res_county",
            "wrk_state_code", "wrk_county_code", "wrk_state", "wrk_county",
            "workers", "moe",
        ],
        estimate=5_800_000,
    ),
]
"""All supported ACS Comm Flow products."""
# fmt: on


class Commuters(FetchADRIO[np.int64, np.int64]):
    """
    Loads data from the US Census Bureau's ACS Commuting Flows product.
    This product uses answers to the American Community Survey over a five year period
    to estimate the number of workers aggregated by where they live and where they work.
    It is a useful estimate of regular commuting activity between locations.

    The product aggregates to the US-County-equivalent granularity, so this ADRIO
    can work with county or state scopes. Because the data are presented using
    FIPS codes, we must be certain to use a compatible scope -- therefore the
    data vintage loaded by this ADRIO is based on the geo scope year and not the
    simulation time frame. Available data years are nominally 2010, 2015, and 2020,
    however note that the 2020 data year was compiled using 2022 geography.

    The result is an NxN matrix of integers, with residency location on the first axis
    and work location on the second axis.

    Parameters
    ----------
    fix_missing :
        The method to use to fix missing values. Missing values are common in this dataset,
        which simply omits pairs of locations for which there were no recorded workers.
        Therefore the default is to fill with zero.

    See Also
    --------
    The [ACS Commuting Flows documentation](https://www.census.gov/topics/employment/commuting/guidance/flows.html)
    from the US Census.
    """  # noqa: E501

    _fix_missing: Fill[np.int64]
    """The method to use to fix missing values."""

    def __init__(
        self,
        *,
        fix_missing: Fill[np.int64] | int | Callable[[], int] | Literal[False] = 0,
    ):
        try:
            self._fix_missing = Fill.of_int64(fix_missing)
        except ValueError:
            raise ValueError("Invalid value for `fix_missing`")

    @property
    @override
    def result_format(self) -> ResultFormat:
        return ResultFormat(shape=Shapes.NxN, dtype=np.int64)

    @override
    def validate_result(self, context: Context, result: NDArray) -> None:
        adrio_validate_pipe(
            self,
            context,
            result,
            validate_numpy(),
            validate_shape(self.result_format.shape.to_tuple(context.dim)),
            validate_dtype(self.result_format.dtype),
            validate_values_in_range(0, None),
        )

    def _get_config(self, context: Context) -> tuple[_Config, StateScope | CountyScope]:
        scope = context.scope
        if not isinstance(scope, StateScope | CountyScope):
            err = "US State or County geo scope required."
            raise ADRIOContextError(self, context, err)

        config = next((x for x in _CONFIG if x.geo_year == scope.year), None)
        if config is None:
            all_years = [str(x.geo_year) for x in _CONFIG]
            err = (
                "Commuters loads data according to your geo scope year "
                "because it is sensitive to changes in geography. "
                "Data is only available for these geo years: "
                f"{','.join(all_years)}"
            )
            raise ADRIOContextError(self, context, err)

        return config, scope

    @override
    def estimate_data(self) -> DataEstimate:
        config, _ = self._get_config(self.context)
        in_cache = check_file_in_cache(config.cache_path)
        total_bytes = config.estimate
        new_bytes = total_bytes if not in_cache else 0
        return AvailableDataEstimate(
            name=self.class_name,
            cache_key=config.cache_key,
            new_network_bytes=new_bytes,
            new_cache_bytes=new_bytes,
            total_cache_bytes=total_bytes,
            max_bandwidth=None,
        )

    @override
    def validate_context(self, context: Context) -> None:
        self._get_config(context)

    @override
    def _fetch(self, context: Context) -> pd.DataFrame:
        config, scope = self._get_config(context)

        # Get and read data file.
        try:
            commuter_file = load_or_fetch_url(config.url, config.cache_path)
        except Exception as e:
            raise ADRIOCommunicationError(self, context) from e

        data_df = pd.read_excel(
            commuter_file,
            header=config.header,
            skipfooter=config.footer,
            names=config.cols,
            usecols=[
                "res_state_code",
                "res_county_code",
                "wrk_state_code",
                "wrk_county_code",
                "workers",
            ],
            dtype={
                "res_state_code": str,
                "wrk_state_code": str,
                "res_county_code": str,
                "wrk_county_code": str,
                "workers": np.int64,
            },
        )

        # Filter out destinations which are not US states and fix two-digit state codes.
        data_df = data_df.loc[data_df["wrk_state_code"].str.startswith("0", na=False)]
        data_df = data_df.assign(wrk_state_code=data_df["wrk_state_code"].str.slice(1))

        # Reformat to combine geoid columns according to result granularity.
        if scope.granularity == "state":
            geoid_src = data_df["res_state_code"]
            geoid_dst = data_df["wrk_state_code"]
        else:
            geoid_src = data_df["res_state_code"] + data_df["res_county_code"]
            geoid_dst = data_df["wrk_state_code"] + data_df["wrk_county_code"]
        data_df = pd.DataFrame(
            {
                "geoid_src": geoid_src,
                "geoid_dst": geoid_dst,
                "value": data_df["workers"],
            }
        )

        # Filter for the geographies in our scope.
        src_in = data_df["geoid_src"].isin(context.scope.node_ids)
        dst_in = data_df["geoid_dst"].isin(context.scope.node_ids)
        data_df = data_df.loc[src_in & dst_in]

        # Aggregate results if our result granularity requires it. Fix indexing.
        if scope.granularity == "state":
            data_df = data_df.groupby(["geoid_src", "geoid_dst"]).sum().reset_index()
        else:
            data_df = data_df.reset_index(drop=True)
        return data_df

    @override
    def _process(self, context: Context, data_df: pd.DataFrame) -> PipelineResult:
        pipeline = DataPipeline(
            axes=(
                PivotAxis("geoid_src", context.scope.node_ids),
                PivotAxis("geoid_dst", context.scope.node_ids),
            ),
            ndims=2,
            dtype=self.result_format.dtype.type,
            rng=context,
        ).finalize(self._fix_missing)
        return pipeline(data_df)
