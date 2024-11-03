from datetime import date as datetype
from datetime import timedelta
from io import BytesIO
from pathlib import Path
from typing import Generator, Literal

import numpy as np
import rasterio.io as rio
from dateutil.relativedelta import relativedelta
from numpy.typing import NDArray
from typing_extensions import override

from epymorph.adrio.adrio import Adrio, ProgressCallback, adrio_cache
from epymorph.cache import check_file_in_cache, load_or_fetch_url, module_cache_path
from epymorph.data_shape import Shapes
from epymorph.data_type import CentroidType
from epymorph.data_usage import AvailableDataEstimate, DataEstimate
from epymorph.error import DataResourceException
from epymorph.geography.scope import GeoScope
from epymorph.geography.us_census import CensusScope
from epymorph.geography.us_geography import STATE
from epymorph.geography.us_tiger import CacheEstimate
from epymorph.simulation import AttributeDef
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


def _validate_scope(scope: GeoScope) -> CensusScope:
    state_fips = list(STATE.truncate_unique(scope.node_ids))
    excluded_fips = ["72", "02", "15"]

    # require census scope for raster values
    if not isinstance(scope, CensusScope):
        msg = "Census scope is required for PRISM attributes."
        raise DataResourceException(msg)

    # scope cannot be in Puerto Rico, Alaska, or Hawaii
    if any(state in excluded_fips for state in state_fips):
        msg = (
            "Alaska, Hawaii, and Puerto Rico cannot be evaluated for PRISM "
            "attributes. Please enter a geoid within the 48 contiguous states."
        )
        raise DataResourceException(msg)
    return scope


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


@adrio_cache
class Precipitation(Adrio[np.float64]):
    """
    Creates an TxN matrix of floats representing the amount of precipitation in an area,
    represented in millimeters (mm).
    """

    date_range: TimeFrame

    requirements = [AttributeDef("centroid", type=CentroidType, shape=Shapes.N)]
    """
    A centroid, or an array of centroids, is required to fetch for a specific point of
    data.
    """

    def __init__(self, date_range: TimeFrame):
        """
        Initializes the precipitation matrix with the date range.

        Parameters
        ----------
        date_range : TimeFrame
            The range of dates to fetch precipitation data for.
        """
        self.date_range = _validate_dates(date_range)

    def estimate_data(self) -> DataEstimate:
        file_size = 1_200_000  # no significant change in size, average to about 1.2MB
        est = _estimate_prism(self, file_size, self.date_range, "ppt")
        return est

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        scope = self.scope
        scope = _validate_scope(scope)
        centroids = self.data("centroid")
        raster_vals = _make_centroid_strategy_adrio(
            "ppt", self.date_range, centroids, self.progress
        )
        return raster_vals


class DewPoint(Adrio[np.float64]):
    """
    Creates an TxN matrix of floats representing the dew point temperature in an area,
    represented in degrees Celsius (°C).
    """

    date_range: TimeFrame

    requirements = [AttributeDef("centroid", type=CentroidType, shape=Shapes.N)]
    """
    A centroid, or an array of centroids, is required to fetch for a specific point of
    data.
    """

    def __init__(self, date_range: TimeFrame):
        """
        Initializes the dew point temperature matrix with the date range.

        Parameters
        ----------
        date_range : TimeFrame
            The range of dates to fetch dew point temperature data for.
        """
        self.date_range = _validate_dates(date_range)

    def estimate_data(self) -> DataEstimate:
        year = self.date_range.end_date.year

        # file sizes are larger after the year 2020
        if year > 2020:
            file_size = 1_800_000  # average to 1.8MB after 2020
        else:
            file_size = 1_400_000  # average to 1.4MB 2020 and before
        return _estimate_prism(self, file_size, self.date_range, "tdmean")

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        scope = self.scope
        scope = _validate_scope(scope)
        centroids = self.data("centroid")
        raster_vals = _make_centroid_strategy_adrio(
            "tdmean", self.date_range, centroids, self.progress
        )
        return raster_vals


class Temperature(Adrio[np.float64]):
    """
    Creates an TxN matrix of floats representing the temperature in an area, represented
    in degrees Celsius (°C).
    """

    date_range: TimeFrame

    requirements = [AttributeDef("centroid", type=CentroidType, shape=Shapes.N)]
    """
    A centroid, or array of centroids, is required to fetch for a specific point of
    data.
    """

    TemperatureType = Literal["Minimum", "Mean", "Maximum"]

    temp_variables: dict[TemperatureType, str] = {
        "Minimum": "tmin",
        "Mean": "tmean",
        "Maximum": "tmax",
    }

    temp_var: TemperatureType

    def __init__(self, date_range: TimeFrame, temp_var: TemperatureType):
        """
        Initializes the temperature matrix with the date range and the statistical
        measure for the temperature.

        Parameters
        ----------
        date_range : TimeFrame
            The range of dates to fetch precipitation data for.
        temp_var : TemperatureType
            The measure of the temperature for a single date (options: 'Minimum',
            'Mean', 'Maximum').
        """
        self.temp_var = temp_var
        self.date_range = _validate_dates(date_range)

    def estimate_data(self) -> DataEstimate:
        year = self.date_range.end_date.year
        temp_var = self.temp_variables[self.temp_var]

        # file sizes are larger after the year 2020
        if year > 2020:
            file_size = 1_700_000  # average to 1.7MB after 2020
        else:
            file_size = 1_400_000  # average to 1.4MB 2020 and before
        return _estimate_prism(self, file_size, self.date_range, temp_var)

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        scope = self.scope
        scope = _validate_scope(scope)
        temp_var = self.temp_variables[self.temp_var]
        centroids = self.data("centroid")
        raster_vals = _make_centroid_strategy_adrio(
            temp_var, self.date_range, centroids, self.progress
        )

        return raster_vals


class VaporPressureDeficit(Adrio[np.float64]):
    """
    Creates an TxN matrix of floats representing the vapor pressure deficit in an area,
    represented in hectopascals (hPa).
    """

    date_range: TimeFrame

    requirements = [AttributeDef("centroid", type=CentroidType, shape=Shapes.N)]
    """
    A centroid, or array of centroids, is required to fetch for a specific point of
    data.
    """

    VPDType = Literal["Minimum", "Maximum"]

    vpd_variables: dict[VPDType, str] = {"Minimum": "vpdmin", "Maximum": "vpdmax"}

    vpd_var: VPDType

    def __init__(self, date_range: TimeFrame, vpd_var: VPDType):
        """
        Initializes the vapor pressure deficit matrix with the date range and the
        statistical measure for the vapor pressure deficit.

        Parameters
        ----------
        date_range : TimeFrame
            The range of dates to fetch precipitation data for.
        vpd_var : VPDType
            The measure of the vapor pressure deficit for a single date
            (options: 'Minimum', 'Maximum').
        """
        self.vpd_var = vpd_var
        self.date_range = _validate_dates(date_range)

    def estimate_data(self) -> DataEstimate:
        year = self.date_range.end_date.year

        # file sizes are larger after the year 2020
        if year > 2020:
            file_size = 1_700_000  # average to 1.7MB after 2020
        else:
            file_size = 1_300_000  # average to 1.3MB 2020 and before
        return _estimate_prism(self, file_size, self.date_range, self.vpd_var)

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        scope = self.scope
        scope = _validate_scope(scope)
        vpd_var = self.vpd_variables[self.vpd_var]
        centroids = self.data("centroid")
        raster_vals = _make_centroid_strategy_adrio(
            vpd_var, self.date_range, centroids, self.progress
        )
        return raster_vals
