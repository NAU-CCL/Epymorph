"""ADRIOs that access Meta's Data For Good Movement Distribution Datasets for daily
movement patterns."""

import calendar
from abc import ABC
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from hdx.api.configuration import Configuration
from hdx.data.dataset import Dataset
from numpy.typing import NDArray
from typing_extensions import override

from epymorph.adrio.adrio import (
    ADRIOContextError,
    ADRIOError,
    FetchADRIO,
    ResultFormat,
    range_mask_fn,
)
from epymorph.adrio.processing import PipelineResult
from epymorph.cache import check_file_in_cache, load_or_fetch_url, module_cache_path
from epymorph.data_shape import Shapes
from epymorph.data_usage import AvailableDataEstimate, DataEstimate
from epymorph.error import DataResourceError
from epymorph.geography.scope import GeoScope
from epymorph.geography.us_census import CensusScope, CountyScope
from epymorph.geography.us_geography import STATE
from epymorph.geography.us_tiger import CacheEstimate, get_counties, get_states
from epymorph.simulation import Context
from epymorph.time import DateRange, TimeFrame

_MOVEMENT_DIST_CACHE_PATH = module_cache_path(__name__)

STATE_ABBRS = [
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "DC",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
]

_STATE_TO_NUM = {abbr: i + 1 for i, abbr in enumerate(STATE_ABBRS)}

_INT_TO_MONTH = {i: month for i, month in enumerate(calendar.month_name) if month}

_MONTH_TO_FILE = {
    "October 2024": "movement-distribution-data-for-good-at-meta_2024-10-01_2024-11-01_csv",
    "November 2024": "1922039342088483_2024-11-01_2024-11-16_csv",
    "December 2024": "1922039342088483_2024-12-01_2024-12-16_csv",
    "January 2025": "movement-distribution-data-for-good-at-meta_2025-01-01_2025-02-01_csv",
    "February 2025": "1922039342088483_2025-02-01_2025-02-16_csv",
}

# create a configuration for the HDX Python API
# reading from the live site and not staging changes and only using GET requests
Configuration.create(
    hdx_site="prod",
    user_agent="EpiMoRPH_MovementDistributionADRIO",
    hdx_read_only=True,
)
DATASET = Dataset.read_from_hdx("movement-distribution")


def _validate_scope(scope: GeoScope) -> CensusScope:
    if not isinstance(scope, CensusScope):
        msg = "Census scope is required for Movement Distribution attributes."
        raise DataResourceError(msg)
    return scope


def _validate_dates(date_range: TimeFrame) -> TimeFrame:
    # from the hdx api, get the valid dataset starting date and ending date
    time_period = DATASET.get_time_period()  # type: ignore

    start_limit = time_period["startdate"].date()
    end_limit = time_period["enddate"].date()

    # if the current date range exceeds the start or end limits of the dataset
    if date_range.start_date < start_limit or date_range.end_date > end_limit:
        msg = (
            "Given date range is out of Movement Distribution scope, please enter dates"
            f" between {start_limit} and {end_limit}"
        )
        raise DataResourceError(msg)
    return date_range


class _MovementADRIO(FetchADRIO[np.float64, np.float64], ABC):
    _override_time_frame: TimeFrame | None
    """An override time frame for which to fetch data.
    If None, the simulation time frame will be used."""

    def __init__(self, time_frame: TimeFrame | None = None):
        self._override_time_frame = time_frame

    @property
    def data_time_frame(self) -> TimeFrame:
        """The time frame for which to fetch data."""
        return self._override_time_frame or self.time_frame


def _estimate_distribution(
    adrio_instance: _MovementADRIO, file_size: int, date_range: TimeFrame
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


class Distribution(_MovementADRIO):
    """
    Creates an TxN matrix of movers and the amount that move within a range of distance
    away from their typical home location.
    """

    time_period = DATASET.get_time_period()  # type: ignore

    start_limit = time_period["startdate"].date()
    end_limit = time_period["enddate"].date()

    _TIME_RANGE = DateRange(start_limit, end_limit)
    """The time range over which values are available."""

    _NUM_ATTRIBUTES = 4

    _RESULT_FORMAT = ResultFormat(
        shape=Shapes.TxNxA,
        value_dtype=np.float64,
        validation=range_mask_fn(minimum=np.float64(0.0), maximum=np.float64(1.0)),
        is_date_value=False,
    )

    def __init__(self, time_frame: TimeFrame | None = None):
        """
        Initializes the time frame for the movement distribution data.

        Parameters
        ----------
        time_frame : int, optional
            The year for the movement distribution data.
            Defaults to the year in which the simulation time frame starts.

        """
        super().__init__(time_frame)

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
            raise DataResourceError(
                "Unable to fetch Movement Distribution Maps data."
            ) from e

        # return files as a pandas dataframe
        return pd.DataFrame({"files": files, "formatted_dates": formatted_dates})

    def _process(
        self,
        context: Context,
        data_df: pd.DataFrame,
    ) -> PipelineResult[np.float64]:
        # set up the categories to search for
        dist_categories = ["0", "(0, 10)", "[10, 100)", "100+"]

        # map the distribution range to index for the results
        category_index = {cat: i for i, cat in enumerate(dist_categories)}
        daily_rows = []
        missing_files = []

        scope = _validate_scope(context.scope)
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
                message=f"The following files were not found in the ZIP archives for {{adrio_name}}:\n{missing_str}",
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

    # estimate the amount of data that will be downloaded
    def estimate_data(self) -> DataEstimate:
        # files from before October 2024 average to 358.8 MB
        if self.data_time_frame.end_date < date(2024, 10, 1):
            file_size = 358_764_000
        # files from October 2024 and after average to 91.82MB
        else:
            file_size = 91_820_000
        est = _estimate_distribution(self, file_size, self.data_time_frame)
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
        scope = context.scope
        time_frame = context.time_frame

        if not isinstance(scope, CountyScope):
            raise ADRIOContextError(self, context, "Scope must be a CountyScope.")

        if len(scope.node_ids) == 0:
            raise ADRIOContextError(
                self, context, "Scope must include at least one county."
            )

        if time_frame.start_date > time_frame.end_date:
            raise ADRIOContextError(
                self, context, "Time frame must have a valid start and end date."
            )
