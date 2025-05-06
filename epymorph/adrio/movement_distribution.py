"""ADRIOs that access Meta's Data For Good Movement Distribution Datasets for daily
movement patterns."""

import calendar
from abc import ABC
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import cast
from zipfile import ZipFile

import numpy as np
import pandas as pd
from hdx.api.configuration import Configuration
from hdx.data.dataset import Dataset
from numpy.typing import NDArray
from typing_extensions import override

from epymorph.adrio.adrio import (
    ADRIOCommunicationError,
    ADRIOContextError,
    ADRIOError,
    FetchADRIO,
    ResultFormat,
    range_mask_fn,
    validate_time_frame,
)
from epymorph.adrio.processing import PipelineResult
from epymorph.cache import check_file_in_cache, load_or_fetch_url, module_cache_path
from epymorph.data_shape import Shapes
from epymorph.data_usage import AvailableDataEstimate, DataEstimate
from epymorph.geography.us_census import CensusScope, CountyScope
from epymorph.geography.us_geography import STATE
from epymorph.geography.us_tiger import CacheEstimate, get_counties, get_states
from epymorph.simulation import Context
from epymorph.time import DateRange, TimeFrame

_MOVEMENT_DIST_CACHE_PATH = module_cache_path(__name__)

_STATE_TO_NUM = {
    "AL": 1,
    "AK": 2,
    "AZ": 3,
    "AR": 4,
    "CA": 5,
    "CO": 6,
    "CT": 7,
    "DE": 8,
    "DC": 9,
    "FL": 10,
    "GA": 11,
    "HI": 12,
    "ID": 13,
    "IL": 14,
    "IN": 15,
    "IA": 16,
    "KS": 17,
    "KY": 18,
    "LA": 19,
    "ME": 20,
    "MD": 21,
    "MA": 22,
    "MI": 23,
    "MN": 24,
    "MS": 25,
    "MO": 26,
    "MT": 27,
    "NE": 28,
    "NV": 29,
    "NH": 30,
    "NJ": 31,
    "NM": 32,
    "NY": 33,
    "NC": 34,
    "ND": 35,
    "OH": 36,
    "OK": 37,
    "OR": 38,
    "PA": 39,
    "RI": 40,
    "SC": 41,
    "SD": 42,
    "TN": 43,
    "TX": 44,
    "UT": 45,
    "VT": 46,
    "VA": 47,
    "WA": 48,
    "WV": 49,
    "WI": 50,
    "WY": 51,
}
"""State to numeric code for GADM polygon codes."""

_INT_TO_MONTH = {i: month for i, month in enumerate(calendar.month_name) if month}

_MONTH_TO_FILE = {
    "October 2024": (
        "movement-distribution-data-for-good-at-meta_2024-10-01_2024-11-01_csv"
    ),
    "November 2024": "1922039342088483_2024-11-01_2024-11-16_csv",
    "December 2024": "1922039342088483_2024-12-01_2024-12-16_csv",
    "January 2025": (
        "movement-distribution-data-for-good-at-meta_2025-01-01_2025-02-01_csv"
    ),
    "February 2025": "1922039342088483_2025-02-01_2025-02-16_csv",
}
"""Folder paths for months that do not follow the typical pattern of the ZIP files"""


Configuration.create(
    hdx_site="prod",
    user_agent="EpiMoRPH_MovementDistributionADRIO",
    hdx_read_only=True,
)
"""
Creates a configuration for the HDX Python API, reading from the live site,
not staging changes. Only uses GET requests and reads from the datasets."""

DATASET = Dataset.read_from_hdx("movement-distribution")
"""Gets the Movement Distribution datasets from the HDX Python API

API Data Source: Humanitarian Data Exchange (HDX)
API Ownership: HDX / UNOCHA (United Nations Office for the Coordination of Humanitarian 
Affairs)
GitHub Repository: https://github.com/OCHA-DAP/hdx-python-api
"""


class Distribution(FetchADRIO[np.float64, np.float64], ABC):
    """
    Creates an TxNxA matrix of movers and the fraction that move within a range of
    distance away from their typical home location. The four ranges go in order from
    moving 0 and 1km, between 0 and 10 km, between 10 and 100 km, and moving anywhere
    above 100km from the home location.
    """

    time_period = DATASET.get_time_period()  # type: ignore
    """
    The available dates of the dataset changes irregularly as new files are not added
    on a consistent basis. This function gets the time frame directly from the dataset.
    """

    start_limit = time_period["startdate"].date()
    """Starting date for Movement Distribution."""

    end_limit = time_period["enddate"].date()
    """Ending date for Movement Distribution."""

    _TIME_RANGE = DateRange(start_limit, end_limit)
    """The time range over which values are available."""

    _NUM_ATTRIBUTES = 4
    """The number of distance range categories: (0, (0, 10), [10, 100), 100+)"""

    _RESULT_FORMAT = ResultFormat(
        shape=Shapes.TxNxA,
        value_dtype=np.float64,
        validation=range_mask_fn(minimum=np.float64(0.0), maximum=np.float64(1.0)),
        is_date_value=False,
    )

    def _fetch(self, context: Context) -> pd.DataFrame:
        start_date = context.time_frame.start_date
        end_date = context.time_frame.end_date

        # format the month for fetching the correct file
        formatted_dates = []

        # get the current starting date
        current = start_date.replace(day=1)

        # move through until the end date
        while current <= end_date:
            # format the month and year as a string and add to a list
            formatted_dates.append(f"{_INT_TO_MONTH[current.month]} {current.year}")

            # increment to the first day of the next month

            # if currently working in december, move to the next year
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)

            # otherwise, just increment to the next month
            else:
                current = current.replace(month=current.month + 1)

        # read from the HDX API for the dataforgood dataset
        # owned by the Humanitarian Data Exchange
        dataset = DATASET

        # get the files from the dataset
        resources = dataset.get_resources()  # type: ignore
        url_list = []

        # for each formatted month, find the file
        for month in formatted_dates:
            for resource in resources:
                # match by description (Month Year)
                if resource.get("description", "").strip() == month:
                    # add the url for that month
                    url_list.append(resource["alt_url"])
                    break

        # download the file/s
        try:
            # rename the file for cache since they are all loaded as "download"
            files = [
                load_or_fetch_url(u, _MOVEMENT_DIST_CACHE_PATH / formatted_date)
                for u, formatted_date in zip(url_list, formatted_dates)
            ]
        except Exception as e:
            raise ADRIOCommunicationError(self, context) from e

        # return files as a pandas dataframe
        return pd.DataFrame({"files": files, "formatted_dates": formatted_dates})

    def _process(
        self,
        context: Context,
        data_df: pd.DataFrame,
    ) -> PipelineResult[np.float64]:
        scope = cast(CensusScope, context.scope)
        # set up the categories to search for
        dist_categories = ["0", "(0, 10)", "[10, 100)", "100+"]

        # map the distribution range to index for the results
        category_index = {cat: i for i, cat in enumerate(dist_categories)}
        daily_rows = []
        missing_files = []

        start_date = context.time_frame.start_date
        end_date = context.time_frame.end_date

        # set up county and state variables
        state_codes = get_states(scope.year).state_fips_to_code
        county_codes = get_counties(scope.year).county_fips_to_name

        # move through each zip file
        dates_by_month = defaultdict(list)
        formatted_dates_set = set()

        # move through all of the dates
        current = start_date
        while current <= end_date:
            # add the formatted month to the dictionary and regular set
            key = current.strftime("%B %Y")
            dates_by_month[key].append(current)
            formatted_dates_set.add(key)
            current += timedelta(days=1)

        # sort formatted dates chronologically
        formatted_dates = data_df["formatted_dates"].tolist()
        file_paths = data_df["files"].tolist()

        # iterate over files and their corresponding dates
        for file_path, formatted_date in zip(file_paths, formatted_dates):
            # open the zip file
            with ZipFile(file_path) as zf:
                for current_day in dates_by_month.get(formatted_date, []):
                    month_key = current_day.strftime("%B %Y")

                    # get any outlier months here
                    if month_key in _MONTH_TO_FILE:
                        folder = _MONTH_TO_FILE[month_key]
                        csv_name = f"{folder}/1922039342088483_{current_day}.csv"
                    else:
                        csv_name = f"1922039342088483_{current_day}.csv"

                    # if there are any missing files, iterate through and
                    # find every missing file for further analysis
                    if csv_name not in zf.namelist():
                        missing_files.append((formatted_date, csv_name))
                        continue

                    # open the current csv as a pandas dataframe
                    with zf.open(csv_name) as file:
                        dist_df = pd.read_csv(file)

                    # for each geoid in the scope
                    for geoid in scope.node_ids:
                        # get the state abbreviation
                        state = STATE.truncate(geoid)
                        state_abbr = state_codes.get(state, "")

                        # get the county name
                        county_name = county_codes.get(geoid, "")[:-4]

                        # translate into the GADM format
                        gadm_state = f"USA.{_STATE_TO_NUM[state_abbr]}."

                        # look for the correct state and county
                        found_state = dist_df[
                            dist_df["gadm_id"].str.startswith(gadm_state)
                        ]
                        found_county = found_state[
                            found_state["gadm_name"] == county_name
                        ]

                        # organize the numpy array rows
                        row_vector = [0.0] * len(dist_categories)

                        # go through each found row
                        for _, row in found_county.iterrows():
                            # select the distance range and the fraction from it
                            cat = row["home_to_ping_distance_category"]
                            frac = row["distance_category_ping_fraction"]

                            # if the cells are not empty
                            if pd.notna(cat) and pd.notna(frac):
                                # enter the fraction into the array accordingly
                                row_vector[category_index[cat]] = frac

                        daily_rows.append(row_vector)

        # if there are any missing files, print out any/all files not in the ZIP file
        if missing_files:
            missing_str = "\n".join(
                f"Missing file for {date}: {name}" for date, name in missing_files
            )
            raise ADRIOError(
                adrio=self,
                context=context,
                message=(
                    "The following files were not found in the ZIP archives for "
                    f"{{adrio_name}}:\n{missing_str}"
                ),
            )

        # organize and shape by nodes and days
        num_days = len(daily_rows) // len(scope.node_ids)
        array = np.array(daily_rows, dtype=np.float64).reshape(
            num_days, len(scope.node_ids), len(dist_categories)
        )
        return PipelineResult(
            value=array,
            issues={},
        )

    def estimate_data(self) -> DataEstimate:
        est = _estimate_distribution(self, self.context.time_frame)
        return est

    @override
    def validate_result(
        self,
        context: Context,
        result: NDArray[np.float64],
        *,
        expected_shape: tuple[int, ...] | None = None,
    ) -> None:
        time_series = self._TIME_RANGE.overlap_or_raise(context.time_frame)
        shp = (len(time_series), context.scope.nodes, self._NUM_ATTRIBUTES)
        super().validate_result(context, result, expected_shape=shp)

    @property
    @override
    def result_format(self) -> ResultFormat[np.float64]:
        return self._RESULT_FORMAT

    def validate_context(self, context: Context) -> None:
        if not isinstance(context.scope, CountyScope):
            raise ADRIOContextError(self, context, "Scope must be a CountyScope.")
        if len(context.scope.node_ids) == 0:
            raise ADRIOContextError(
                self, context, "Scope must include at least one county."
            )
        validate_time_frame(self, context, self._TIME_RANGE)


def _estimate_distribution(
    adrio_instance: Distribution, date_range: TimeFrame
) -> DataEstimate:
    """
    Calculate estimates for downloading Movement Distribution files.
    """
    # set cache size variables
    missing_cache_size = 0
    total_cache_size = 0

    # get the months that will be loaded and check in cache
    formatted_dates = []

    # set the current starting date
    current = date_range.start_date.replace(day=1)

    # iterate through each day
    while current <= date_range.end_date:
        # format the current month variable
        file = f"{_INT_TO_MONTH[current.month]} {current.year}"

        # add the formatted date to the list of months
        formatted_dates.append(file)

        # check for cache size, files before October 2024 are much larger
        if current < date(2024, 10, 1):
            # add to the total cache
            total_cache_size += 358_764_000
            # if the file is not in the cache, add to the missing cache size
            if not check_file_in_cache(_MOVEMENT_DIST_CACHE_PATH / Path(file).name):
                missing_cache_size += 358_764_000
        else:
            # add to the total cache
            total_cache_size += 91_820_000
            # if the file is not in the cache, add to the missing cache size
            if not check_file_in_cache(_MOVEMENT_DIST_CACHE_PATH / Path(file).name):
                missing_cache_size += 91_820_000

        # move to the first of next month

        # if currently checking for december, move to the next year
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        # otherwise, just move to the next month
        else:
            current = current.replace(month=current.month + 1)

    # calculate the cache estimate
    est = CacheEstimate(
        total_cache_size=total_cache_size,
        missing_cache_size=missing_cache_size,
    )

    key = f"distribution:{date_range}"
    return AvailableDataEstimate(
        name=adrio_instance.class_name,
        cache_key=key,
        new_network_bytes=est.missing_cache_size,
        new_cache_bytes=est.missing_cache_size,
        total_cache_bytes=est.total_cache_size,
        max_bandwidth=None,
    )
