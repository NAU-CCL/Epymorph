from pathlib import Path
from typing import Callable, Literal, Mapping, NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing_extensions import override

from epymorph.adrio.adrio import (
    ADRIOPrototype,
    Fill,
    ProcessResult,
    process_nxn,
    range_mask_fn,
)
from epymorph.cache import check_file_in_cache, load_or_fetch_url, module_cache_path
from epymorph.data_usage import AvailableDataEstimate, DataEstimate
from epymorph.error import DataResourceError
from epymorph.geography.us_census import CountyScope, StateScope
from epymorph.simulation import Context
from epymorph.time import DateRange, iso8601

_COMMFLOWS_CACHE_PATH = module_cache_path(__name__)

_ValidYear = Literal[2010, 2015, 2020]
_ValidGranularity = Literal["state", "county"]


class _Config(NamedTuple):
    year: _ValidYear
    url: str
    header: int
    footer: int
    cols: list[str]
    estimate: int

    @property
    def cache_path(self) -> Path:
        return _COMMFLOWS_CACHE_PATH / f"{self.year}.xlsx"


_CONFIG: Mapping[_ValidYear, _Config] = {
    2010: _Config(
        year=2010,
        url="https://www2.census.gov/programs-surveys/demo/tables/metro-micro/2010/commuting-employment-2010/table1.xlsx",
        header=4,
        footer=3,
        cols=[
            "res_state_code",
            "res_county_code",
            "wrk_state_code",
            "wrk_county_code",
            "workers",
            "moe",
            "res_state",
            "res_county",
            "wrk_state",
            "wrk_county",
        ],
        estimate=7_200_000,
    ),
    2015: _Config(
        year=2015,
        url="https://www2.census.gov/programs-surveys/demo/tables/metro-micro/2015/commuting-flows-2015/table1.xlsx",
        header=6,
        footer=2,
        cols=[
            "res_state_code",
            "res_county_code",
            "res_state",
            "res_county",
            "wrk_state_code",
            "wrk_county_code",
            "wrk_state",
            "wrk_county",
            "workers",
            "moe",
        ],
        estimate=6_700_000,
    ),
    2020: _Config(
        year=2020,
        url="https://www2.census.gov/programs-surveys/demo/tables/metro-micro/2020/commuting-flows-2020/table1.xlsx",
        header=7,
        footer=4,
        cols=[
            "res_state_code",
            "res_county_code",
            "res_state",
            "res_county",
            "wrk_state_code",
            "wrk_county_code",
            "wrk_state",
            "wrk_county",
            "workers",
            "moe",
        ],
        estimate=5_800_000,
    ),
}

# TODO: check what's going on with connecticut data
# https://developer.ap.org/ap-elections-api/docs/CT_FIPS_Codes_forPlanningRegions.htm
# Oh no...
# 2020 Comm Flows uses 2022 CT planning regions, which is when they officially changed
# from counties; should we enforce that 2020 Comm Flows uses 2022 geography?
# Is that valid for the other states? Or do we need a wacky Frankenstein geo just for
# 2020 comm flows?
# This is documented in the footer of the spreadsheet:
# "This table uses county equivalents (planning regions) for Connecticut."  # noqa: E501, ERA001


class Commuters(ADRIOPrototype[np.int64]):
    _TIME_RANGE = DateRange(iso8601("2019-12-29"), iso8601("2024-04-21"), step=7)
    _VALUE_RANGE = range_mask_fn(minimum=np.int64(0), maximum=None)

    fix_missing: Fill[np.int64]

    def __init__(
        self,
        *,
        fix_missing: Fill[np.int64] | int | Callable[[], int] | Literal[False] = False,
    ):
        try:
            self.fix_missing = Fill.of_int64(fix_missing)
        except ValueError:
            raise ValueError("Invalid value for `fix_missing`")

    def _validate_context_internal(
        self,
        context: Context,
    ) -> tuple[_ValidGranularity, _ValidYear]:
        scope = context.scope
        if not isinstance(scope, StateScope | CountyScope):
            err = "US State or County geo scope required."
            raise DataResourceError(err)

        year = scope.year
        if year not in [2010, 2015, 2020]:
            err = "Commuters data is only available for 2010, 2015, and 2020 geography."
            raise DataResourceError(err)

        return scope.granularity, year  # type: ignore

    @override
    def estimate_data(self) -> DataEstimate:
        _, year = self._validate_context_internal(self.context)
        config = _CONFIG[year]
        in_cache = check_file_in_cache(config.cache_path)
        total_bytes = config.estimate
        new_bytes = total_bytes if not in_cache else 0
        return AvailableDataEstimate(
            name=self.class_name,
            cache_key=f"commflows:{year}",
            new_network_bytes=new_bytes,
            new_cache_bytes=new_bytes,
            total_cache_bytes=total_bytes,
            max_bandwidth=None,
        )

    @override
    def _validate_context(self, context: Context):
        self._validate_context_internal(context)

    @override
    def _fetch(self, context: Context) -> pd.DataFrame:
        granularity, year = self._validate_context_internal(context)
        config = _CONFIG[year]

        try:
            commuter_file = load_or_fetch_url(config.url, config.cache_path)
        except Exception as e:
            err = "Unable to fetch commuting flows data."
            raise DataResourceError(err) from e

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

        data_df = data_df.loc[data_df["wrk_state_code"].str.startswith("0", na=False)]
        data_df = data_df.assign(wrk_state_code=data_df["wrk_state_code"].str.slice(1))

        if granularity == "state":
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

        src_in = data_df["geoid_src"].isin(context.scope.node_ids)
        dst_in = data_df["geoid_dst"].isin(context.scope.node_ids)
        data_df = data_df.loc[src_in & dst_in]

        if granularity == "state":
            data_df = data_df.groupby(["geoid_src", "geoid_dst"]).sum().reset_index()
        else:
            data_df = data_df.reset_index(drop=True)
        return data_df

    @override
    def _process(self, context: Context, data_df: pd.DataFrame) -> ProcessResult:
        return process_nxn(
            sentinels=[],
            fix_missing=self.fix_missing,
            dtype=np.int64,
            context=context,
            data_df=data_df,
        )

    @override
    def _validate_result(self, context: Context, result: NDArray[np.int64]) -> None:
        # NOTE: validation only checks non-masked values
        if not Commuters._VALUE_RANGE(result).all():
            raise ValueError("invalid values")  # TODO

        expected_shape = (context.scope.nodes, context.scope.nodes)
        if result.shape != expected_shape:
            raise ValueError("invalid shape")  # TODO

        if np.dtype(result.dtype) != np.dtype(np.int64):
            raise ValueError("invalid dtype")  # TODO
