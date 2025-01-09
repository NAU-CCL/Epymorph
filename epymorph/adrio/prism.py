from abc import ABC, abstractmethod
from datetime import date as datetype
from datetime import timedelta
from io import BytesIO
from pathlib import Path
from typing import Generator, Literal
from warnings import warn

import numpy as np
import rasterio.io as rio
from dateutil.relativedelta import relativedelta
from numpy.typing import NDArray

from epymorph.adrio.adrio import Adrio, ProgressCallback, adrio_cache
from epymorph.attribute import AttributeDef
from epymorph.cache import check_file_in_cache, load_or_fetch_url, module_cache_path
from epymorph.data_shape import Shapes
from epymorph.data_type import CentroidType
from epymorph.data_usage import AvailableDataEstimate, DataEstimate
from epymorph.error import DataResourceException
from epymorph.geography.scope import GeoScope
from epymorph.geography.us_census import CensusScope
from epymorph.geography.us_geography import STATE
from epymorph.geography.us_tiger import CacheEstimate
from epymorph.time import TimeFrame

_PRISM_CACHE_PATH = module_cache_path(__name__)


def _generate_file_name(
    attribute: str,
    latest_date: datetype,
    last_completed_month: datetype,
    date: datetype,
) -> tuple[str, str]:
    """
    Generates the url for the given date and climate attribute. Returns a tuple
    of strings with the url and the name of the bil file within the zip file.
    """

    if date.year == latest_date.year and date.month == latest_date.month:
        stability = "early"

    # if it is before the last finished month
    elif date > last_completed_month:
        stability = "provisional"

    # if it is older than 6 completed months
    else:
        stability = "stable"

    # format the date for the url
    formatted_date = date.strftime("%Y%m%d")
    year = date.year

    url = f"https://ftp.prism.oregonstate.edu/daily/{attribute}/{year}/PRISM_{attribute}_{stability}_4kmD2_{formatted_date}_bil.zip"

    bil_name = f"PRISM_{attribute}_{stability}_4kmD2_{formatted_date}_bil.bil"

    return url, bil_name


def _fetch_raster(
    attribute: str, date_range: TimeFrame, progress: ProgressCallback
) -> Generator[BytesIO, None, None]:
    """
    Fetches the raster values at the url with the given attribute and date range.
    """

    # set some date variables with the date_range
    latest_date = datetype.today() - timedelta(days=1)
    first_day = date_range.start_date
    last_day = date_range.end_date

    # create the list of days in date_range
    date_list = [
        first_day + timedelta(days=x) for x in range((last_day - first_day).days + 1)
    ]

    # the stability of PRISM data is defined by date, specified around the 6 month mark
    six_months_ago = datetype.today() + relativedelta(months=-6)
    last_completed_month = six_months_ago.replace(day=1) - timedelta(days=1)

    # for progress tracking
    processing_steps = len(date_list) + 1

    for i, single_date in enumerate(date_list):
        url, bil_name = _generate_file_name(
            attribute, latest_date, last_completed_month, single_date
        )

        # load/fetch the url for the file
        try:
            file = load_or_fetch_url(url, _PRISM_CACHE_PATH / Path(url).name)

        except Exception as e:
            raise DataResourceException("Unable to fetch PRISM data.") from e

        # if the progress isnt None
        if progress is not None:
            # progress by one, increasing percentage done
            progress((i + 1) / processing_steps, None)

        file.name = bil_name

        yield file


def _make_centroid_strategy_adrio(
    attribute: str, date: TimeFrame, centroids: NDArray, progress: ProgressCallback
) -> NDArray[np.float64]:
    """
    Retrieves the raster value at a centroid of a granularity.
    """
    raster_files = _fetch_raster(attribute, date, progress)
    results = []

    # read in each file
    for raster_file in raster_files:
        with rio.ZipMemoryFile(raster_file) as zip_contents:
            with zip_contents.open(raster_file.name) as dataset:
                values = [x[0] for x in dataset.sample(centroids)]

        results.append(values)

    return np.array(results, dtype=np.float64)


def _validate_dates(date_range: TimeFrame) -> TimeFrame:
    latest_date = datetype.today() - timedelta(days=1)
    # PRISM only accounts for dates from 1981 up to yesterday's date
    if date_range.start_date.year < 1981 or latest_date < date_range.end_date:
        msg = (
            "Given date range is out of PRISM scope, please enter dates between "
            f"1981-01-01 and {latest_date}"
        )
        raise DataResourceException(msg)

    return date_range


def _estimate_prism(
    self, file_size: int, date_range: TimeFrame, attribute: str
) -> DataEstimate:
    """
    Calculate estimates for downloading PRISM files.
    """
    est_file_size = file_size
    total_files = date_range.duration_days

    # setup urls as list to check if theyre in the cache

    # setup date variables
    first_day = date_range.start_date
    last_day = date_range.end_date
    latest_date = datetype.today() - timedelta(days=1)
    six_months_ago = datetype.today() + relativedelta(months=-6)
    last_completed_month = six_months_ago.replace(day=1) - timedelta(days=1)
    date_list = [
        first_day + timedelta(days=x) for x in range((last_day - first_day).days + 1)
    ]

    # get url names to check in cache
    urls = [
        _generate_file_name(attribute, latest_date, last_completed_month, day)[0]
        for day in date_list
    ]

    # sum the files needed to download
    missing_files = total_files - sum(
        1 for u in urls if check_file_in_cache(_PRISM_CACHE_PATH / Path(u).name)
    )

    # calculate the cache estimate
    est = CacheEstimate(
        total_cache_size=total_files * est_file_size,
        missing_cache_size=missing_files * est_file_size,
    )

    key = f"prism:{attribute}:{date_range}"
    return AvailableDataEstimate(
        name=self.full_name,
        cache_key=key,
        new_network_bytes=est.missing_cache_size,
        new_cache_bytes=est.missing_cache_size,
        total_cache_bytes=est.total_cache_size,
        max_bandwidth=None,
    )


class _PRISMAdrio(Adrio[np.float64], ABC):
    _override_time_frame: TimeFrame | None
    """An override time frame for which to fetch data.
    If None, the simulation time frame will be used."""

    errors: Literal["raise", "warn", "ignore"]
    """How to handle data errors."""

    def __init__(
        self,
        time_frame: TimeFrame | None = None,
        errors: Literal["raise", "warn", "ignore"] = "raise",
    ):
        self._override_time_frame = time_frame
        self.errors = errors

    @property
    def data_time_frame(self) -> TimeFrame:
        """The time frame for which to fetch data."""
        return self._override_time_frame or self.time_frame

    def _validate_scope(self) -> GeoScope:
        """
        When a scope is a CensusScope, check if there is a given location in Hawaii,
        Alaska, or Puerto Rico.
        """
        scope = self.scope
        if isinstance(scope, CensusScope):
            state_fips = list(STATE.truncate_unique(scope.node_ids))
            excluded_fips = ["72", "02", "15"]
            # scope cannot be in Puerto Rico, Alaska, or Hawaii
            if any(state in excluded_fips for state in state_fips):
                msg = (
                    "Alaska, Hawaii, and Puerto Rico cannot be evaluated for PRISM "
                    "attributes. Please enter a geoid within the 48 contiguous states."
                )
                raise DataResourceException(msg)
        return scope

    @abstractmethod
    def retrieve_prism(self) -> NDArray[np.float64]:
        """
        Allow for the PRISM ADRIOs to retrieve their raster data for their climate
        attribute type.
        """

    def _validate_data(
        self, raster_data: NDArray, centroids: NDArray
    ) -> NDArray[np.float64]:
        """
        Check the fetched data for sentinel values. By default, raise an error on the
        instance of an invalid value. Otherwise, refer to the user parameter
        specification.
        """
        scope = self.scope
        errors = self.errors
        # check for any invalid values, handle error accordingly
        if errors != "ignore":
            (indices,) = np.nonzero(np.any(raster_data == -9999, axis=0))
            if len(indices) > 0:
                # get the points where there are sentinel values
                invalid_centroids = np.unique(centroids[indices])
                invalid_nodes = np.unique(scope.node_ids[indices])

                # correlate the nodes and centroids to show where the error is occurring
                table_title = ["\nGEOID               Centroid"]
                for node, centroid in zip(invalid_nodes, invalid_centroids):
                    table_title.append(f"{str(node).ljust(20)}{centroid}")

                node_table = "\n".join(table_title)

                __cls = self.__class__.__name__

                error_msg = (
                    "\nOne or more of the centroids provided are outside of the "
                    f"geographic bounds defined by PRISM. PRISM {__cls} has not "
                    "returned data for the following nodes and centroids:"
                    f"\n{node_table}"
                    "\n\nThis issue may occur if a centroid is placed in a body of "
                    "water or is located outside of the 48 contiguous United States. "
                    "\n\nThere are a couple of ways to handle this: "
                    "\n1. Adjust the centroids"
                    "\n  - If feasible, adjust the above centroids accordingly to be on"
                    " land, within the 48 adjoining U.S. states."
                    "\n\n2. Error Handling"
                    "\n  - By default, PRISM ADRIOs will default to raise an error when"
                    " a centroid does not return valid data. "
                    "\n  - This setting can be changed by "
                    "setting the errors parameter at the end of the PRISM ADRIO calls "
                    "to any of the following: "
                    "\n\t- `errors='raise'`"
                    "\n\t- `errors='warn'`"
                    "\n\t- `errors='ignore'`"
                )

                if errors == "raise":
                    raise DataResourceException(error_msg)
                elif errors == "warn":
                    warn(error_msg)

        return raster_data

    def evaluate_adrio(self) -> NDArray[np.float64]:
        _validate_dates(self.data_time_frame)
        self._validate_scope()
        raster_vals = self.retrieve_prism()
        raster_vals = self._validate_data(raster_vals, self.data("centroid"))
        return raster_vals


@adrio_cache
class Precipitation(_PRISMAdrio):
    """
    Creates an TxN matrix of floats representing the amount of precipitation in an area,
    represented in millimeters (mm).
    """

    requirements = [AttributeDef("centroid", type=CentroidType, shape=Shapes.N)]
    """
    An array of centroids is required to fetch for a specific point of
    data.
    """

    def __init__(
        self,
        time_frame: TimeFrame | None = None,
        errors: Literal["raise", "warn", "ignore"] = "raise",
    ):
        """
        Creates a precipitation ADRIO.

        Parameters
        ----------
        time_frame : TimeFrame, optional
            The range of dates to fetch precipitation data for.
            Default: the simulation time frame.
        errors : {'raise', 'warn', 'ignore'}, optional
            Error handling for potential out-of-bound centroids.
            - `'raise'`: Raises an error when out-of-bound centroids are given."
            " No resulting matrices will be shown (default).
            - `'warn'`: Issues a warning about invalid centroids, "
            "but displays the resulting matrices.
            - `'ignore'`: Does not display any message concerning invalid centroids"
            " and returns the resulting matrices.
        """
        super().__init__(time_frame, errors)

    def estimate_data(self) -> DataEstimate:
        file_size = 1_200_000  # no significant change in size, average to about 1.2MB
        est = _estimate_prism(self, file_size, self.data_time_frame, "ppt")
        return est

    def retrieve_prism(self) -> NDArray[np.float64]:
        centroids = self.data("centroid")
        raster_vals = _make_centroid_strategy_adrio(
            "ppt", self.data_time_frame, centroids, self.progress
        )
        return raster_vals


@adrio_cache
class DewPoint(_PRISMAdrio):
    """
    Creates an TxN matrix of floats representing the dew point temperature in an area,
    represented in degrees Celsius (°C).
    """

    requirements = [AttributeDef("centroid", type=CentroidType, shape=Shapes.N)]
    """
    An array of centroids is required to fetch for a specific point of
    data.
    """

    def __init__(
        self,
        time_frame: TimeFrame | None = None,
        errors: Literal["raise", "warn", "ignore"] = "raise",
    ):
        """
        Creates a dew point ADRIO.

        Parameters
        ----------
        time_frame : TimeFrame, optional
            The range of dates to fetch dew point temperature data for.
            Default: the simulation time frame.
        errors : {'raise', 'warn', 'ignore'}, optional
            Error handling for potential out-of-bound centroids.
            - `'raise'`: Raises an error when out-of-bound centroids are given."
            " No resulting matrices will be shown (default).
            - `'warn'`: Issues a warning about invalid centroids, "
            "but displays the resulting matrices.
            - `'ignore'`: Does not display any message concerning invalid centroids"
            " and returns the resulting matrices.
        """
        super().__init__(time_frame, errors)

    def estimate_data(self) -> DataEstimate:
        year = self.data_time_frame.end_date.year

        # file sizes are larger after the year 2020
        if year > 2020:
            file_size = 1_800_000  # average to 1.8MB after 2020
        else:
            file_size = 1_400_000  # average to 1.4MB 2020 and before
        return _estimate_prism(self, file_size, self.data_time_frame, "tdmean")

    def retrieve_prism(self) -> NDArray[np.float64]:
        centroids = self.data("centroid")
        raster_vals = _make_centroid_strategy_adrio(
            "tdmean", self.data_time_frame, centroids, self.progress
        )
        return raster_vals


@adrio_cache
class Temperature(_PRISMAdrio):
    """
    Creates an TxN matrix of floats representing the temperature in an area, represented
    in degrees Celsius (°C).
    """

    requirements = [AttributeDef("centroid", type=CentroidType, shape=Shapes.N)]
    """
    An array of centroids is required to fetch for a specific point of
    data.
    """

    TemperatureType = Literal["Minimum", "Mean", "Maximum"]

    temp_variables: dict[TemperatureType, str] = {
        "Minimum": "tmin",
        "Mean": "tmean",
        "Maximum": "tmax",
    }

    temp_var: TemperatureType

    def __init__(
        self,
        temp_var: TemperatureType,
        time_frame: TimeFrame | None = None,
        errors: Literal["raise", "warn", "ignore"] = "raise",
    ):
        """
        Creates a temperature ADRIO for the given statistical measure (min/max/mean).

        Parameters
        ----------
        temp_var : TemperatureType
            The measure of the temperature for a single date (options: 'Minimum',
            'Mean', 'Maximum').
        time_frame : TimeFrame, optional
            The range of dates to fetch precipitation data for.
            Default: the simulation time frame.
        errors : {'raise', 'warn', 'ignore'}, optional
            Error handling for potential out-of-bound centroids.
            - `'raise'`: Raises an error when out-of-bound centroids are given."
            " No resulting matrices will be shown (default).
            - `'warn'`: Issues a warning about invalid centroids, "
            "but displays the resulting matrices.
            - `'ignore'`: Does not display any message concerning invalid centroids"
            " and returns the resulting matrices.
        """
        super().__init__(time_frame, errors)
        self.temp_var = temp_var

    def estimate_data(self) -> DataEstimate:
        year = self.data_time_frame.end_date.year
        temp_var = self.temp_variables[self.temp_var]

        # file sizes are larger after the year 2020
        if year > 2020:
            file_size = 1_700_000  # average to 1.7MB after 2020
        else:
            file_size = 1_400_000  # average to 1.4MB 2020 and before
        return _estimate_prism(self, file_size, self.data_time_frame, temp_var)

    def retrieve_prism(self) -> NDArray[np.float64]:
        temp_var = self.temp_variables[self.temp_var]
        centroids = self.data("centroid")
        raster_vals = _make_centroid_strategy_adrio(
            temp_var, self.data_time_frame, centroids, self.progress
        )
        return raster_vals


@adrio_cache
class VaporPressureDeficit(_PRISMAdrio):
    """
    Creates an TxN matrix of floats representing the vapor pressure deficit in an area,
    represented in hectopascals (hPa).
    """

    requirements = [AttributeDef("centroid", type=CentroidType, shape=Shapes.N)]
    """
    An array of centroids is required to fetch for a specific point of
    data.
    """

    VPDType = Literal["Minimum", "Maximum"]

    vpd_variables: dict[VPDType, str] = {"Minimum": "vpdmin", "Maximum": "vpdmax"}

    vpd_var: VPDType

    def __init__(
        self,
        vpd_var: VPDType,
        time_frame: TimeFrame | None = None,
        errors: Literal["raise", "warn", "ignore"] = "raise",
    ):
        """
        Creates a vapor pressure deficit ADRIO for the given statistical measure
        (min/max).

        Parameters
        ----------
        vpd_var : VPDType
            The measure of the vapor pressure deficit for a single date
            (options: 'Minimum', 'Maximum').
        time_frame : TimeFrame, optional
            The range of dates to fetch precipitation data for.
            Default: the simulation time frame.
        errors : {'raise', 'warn', 'ignore'}, optional
            Error handling for potential out-of-bound centroids.
            - `'raise'`: Raises an error when out-of-bound centroids are given."
            " No resulting matrices will be shown (default).
            - `'warn'`: Issues a warning about invalid centroids, "
            "but displays the resulting matrices.
            - `'ignore'`: Does not display any message concerning invalid centroids"
            " and returns the resulting matrices.
        """
        super().__init__(time_frame, errors)
        self.vpd_var = vpd_var

    def estimate_data(self) -> DataEstimate:
        year = self.data_time_frame.end_date.year

        # file sizes are larger after the year 2020
        if year > 2020:
            file_size = 1_700_000  # average to 1.7MB after 2020
        else:
            file_size = 1_300_000  # average to 1.3MB 2020 and before
        return _estimate_prism(self, file_size, self.data_time_frame, self.vpd_var)

    def retrieve_prism(self) -> NDArray[np.float64]:
        vpd_var = self.vpd_variables[self.vpd_var]
        centroids = self.data("centroid")
        raster_vals = _make_centroid_strategy_adrio(
            vpd_var, self.data_time_frame, centroids, self.progress
        )
        return raster_vals
