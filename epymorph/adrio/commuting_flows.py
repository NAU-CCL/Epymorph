"""ADRIOs that access the US Census ACS Commuting Flows files."""

import numpy as np
from numpy.typing import NDArray
from pandas import read_excel
from typing_extensions import override

from epymorph.adrio.adrio import Adrio
from epymorph.cache import check_file_in_cache, load_or_fetch_url, module_cache_path
from epymorph.data_usage import DataEstimate
from epymorph.error import DataResourceException
from epymorph.geography.scope import GeoScope
from epymorph.geography.us_census import (
    BlockGroupScope,
    CensusScope,
    StateScope,
    StateScopeAll,
    TractScope,
)
from epymorph.geography.us_tiger import CacheEstimate

_COMMFLOWS_CACHE_PATH = module_cache_path(__name__)


def _validate_year(scope: CensusScope):
    year = scope.year
    if year not in [2010, 2015, 2020]:
        # if invalid year is close to a valid year, fetch valid data and notify user
        passed_year = year
        if year in range(2010, 2015):
            year = 2010
        elif year in range(2015, 2020):
            year = 2015
        elif year in range(2020, 2024):
            year = 2020
        else:
            msg = "Invalid year. Commuting data is only available for 2010-2023"
            raise DataResourceException(msg)

        print(
            f"Commuting data cannot be retrieved for {passed_year}, "
            f"fetching {year} data instead."
        )

    return year


def _validate_scope(scope: GeoScope) -> CensusScope:
    if not isinstance(scope, CensusScope):
        msg = "Census scope is required for commuting flows data."
        raise DataResourceException(msg)

    # check for invalid granularity
    if isinstance(scope, TractScope | BlockGroupScope):
        msg = (
            "Commuting data cannot be retrieved for tract "
            "or block group granularities"
        )
        raise DataResourceException(msg)

    year = scope.year
    node_ids = scope.get_node_ids()
    # a discrepancy exists in data for Connecticut counties in 2020 and 2021
    # raise an exception if this data is requested for these years.
    if year in [2020, 2021] and any(
        connecticut_county in node_ids
        for connecticut_county in [
            "09001",
            "09003",
            "09005",
            "09007",
            "09009",
            "09011",
            "09013",
            "09015",
        ]
    ):
        msg = (
            "Commuting flows data cannot be retrieved for Connecticut counties "
            "for years 2020 or 2021."
        )
        raise DataResourceException(msg)

    return scope


class Commuters(Adrio[np.int64]):
    """Makes an ADRIO to retrieve ACS commuting flow data."""

    def estimate_data(self) -> DataEstimate:
        scope = _validate_scope(self.scope)
        year = _validate_year(scope)

        est_file_size = 0

        # file size is dependant on the year
        if year == 2010:
            est_file_size = 7_200_000  # 2010 table1 is 7.2MB
        elif year == 2015:
            est_file_size = 6_700_000  # 2015 table1 is 6.7MB
        elif year == 2020:
            est_file_size = 5_800_000  # 2020 table1 is 5.8MB

        # only one url, just check if in cache or not
        missing_files = (
            0 if check_file_in_cache(_COMMFLOWS_CACHE_PATH / f"{year}.xlsx") else 1
        )

        # cache estimate
        est = CacheEstimate(
            total_cache_size=est_file_size,
            missing_cache_size=missing_files * est_file_size,
        )

        key = f"commflows:{year}"
        return DataEstimate(
            name=self.full_name,
            cache_key=key,
            new_network_bytes=est.missing_cache_size,
            new_cache_bytes=est.missing_cache_size,
            total_cache_bytes=est.total_cache_size,
            max_bandwidth=None,
        )

    @override
    def evaluate_adrio(self) -> NDArray[np.int64]:
        scope = _validate_scope(self.scope)
        year = _validate_year(scope)
        progress = self.progress

        # start progress tracking, +1 for post processing
        processing_steps = 2

        if year != 2010:
            url = f"https://www2.census.gov/programs-surveys/demo/tables/metro-micro/{year}/commuting-flows-{year}/table1.xlsx"

            # organize dataframe column names
            group_fields = ["state_code", "county_code", "state", "county"]

            all_fields = (
                ["res_" + field for field in group_fields]
                + ["wrk_" + field for field in group_fields]
                + ["workers", "moe"]
            )

            header_num = 7

        else:
            url = "https://www2.census.gov/programs-surveys/demo/tables/metro-micro/2010/commuting-employment-2010/table1.xlsx"

            all_fields = [
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
            ]

            header_num = 4

        node_ids = scope.get_node_ids()

        try:
            cache_path = _COMMFLOWS_CACHE_PATH / f"{year}.xlsx"
            commuter_file = load_or_fetch_url(url, cache_path)
        except Exception as e:
            raise DataResourceException("Unable to fetch commuting flows data.") from e

        # increment progress, just one step here
        if progress is not None:
            progress(1 / processing_steps, None)

        # download communter data spreadsheet as a pandas dataframe
        data_df = read_excel(
            commuter_file,
            header=header_num,
            names=all_fields,
            dtype={
                "res_state_code": str,
                "wrk_state_code": str,
                "res_county_code": str,
                "wrk_county_code": str,
            },
        )

        match scope.granularity:
            case "state":
                data_df = data_df.rename(
                    columns={
                        "res_state_code": "res_geoid",
                        "wrk_state_code": "wrk_geoid",
                    }
                )

            case "county":
                data_df["res_geoid"] = (
                    data_df["res_state_code"] + data_df["res_county_code"]
                )
                data_df["wrk_geoid"] = (
                    data_df["wrk_state_code"] + data_df["wrk_county_code"]
                )

            case _:
                raise DataResourceException("Unsupported query.")

        # Filter out GEOIDs that aren't in our scope.
        res_selection = data_df["res_geoid"].isin(node_ids)
        wrk_selection = data_df["wrk_geoid"].isin(["0" + x for x in node_ids])
        data_df = data_df[res_selection & wrk_selection]

        if isinstance(scope, StateScope | StateScopeAll):
            # Data is county level; group and aggregate to get state level
            data_df = (
                data_df.groupby(["res_geoid", "wrk_geoid"])
                .agg({"workers": "sum"})
                .reset_index()
            )

        return (
            data_df.pivot_table(
                index="res_geoid",
                columns="wrk_geoid",
                values="workers",
            )
            .fillna(0)
            .to_numpy(dtype=np.int64)
        )
