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

from epymorph.adrio.adrio import Adrio
from epymorph.cache import load_or_fetch_url, module_cache_path
from epymorph.data_shape import Shapes
from epymorph.data_type import CentroidType
from epymorph.error import DataResourceException
from epymorph.simulation import AttributeDef, TimeFrame

_PRISM_CACHE_PATH = module_cache_path(__name__)


def _fetch_raster(
    attribute: str, date_range: TimeFrame
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

    for single_date in date_list:
        if (
            single_date.year == latest_date.year
            and single_date.month == latest_date.month
        ):
            stability = "early"

        # if it is before the last finished month
        elif single_date > last_completed_month:
            stability = "provisional"

        # if it is older than 6 completed months
        else:
            stability = "stable"

        # format the date for the url
        formatted_date = single_date.strftime("%Y%m%d")
        year = single_date.year

        url = f"https://ftp.prism.oregonstate.edu/daily/{attribute}/{year}/PRISM_{attribute}_{stability}_4kmD2_{formatted_date}_bil.zip"

        # load/fetch the url for the file
        try:
            file = load_or_fetch_url(url, _PRISM_CACHE_PATH / Path(url).name)

        except Exception as e:
            raise DataResourceException("Unable to fetch PRISM data.") from e

        file.name = f"PRISM_{attribute}_{stability}_4kmD2_{formatted_date}_bil.bil"

        yield file


def _make_centroid_strategy_adrio(
    attribute: str, date: TimeFrame, centroids: NDArray
) -> NDArray[np.float64]:
    """
    Retrieves the raster value at a centroid of a granularity.
    """
    raster_files = _fetch_raster(attribute, date)
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


class Precipitation(Adrio[np.float64]):
    """
    Creates an TxN matrix of floats representing the amount of precipitation in an area,
    represented in millimeters (mm).
    """

    date_range: TimeFrame

    requirements = [AttributeDef("centroid", type=CentroidType, shape=Shapes.N)]

    def __init__(self, date_range: TimeFrame):
        self.date_range = _validate_dates(date_range)

    @override
    def evaluate(self) -> NDArray[np.float64]:
        centroids = self.data("centroid")
        raster_vals = _make_centroid_strategy_adrio("ppt", self.date_range, centroids)
        return raster_vals


class DewPoint(Adrio[np.float64]):
    """
    Creates an TxN matrix of floats representing the dew point temperature in an area,
    represented in degrees Celsius (°C).
    """

    date_range: TimeFrame

    requirements = [AttributeDef("centroid", type=CentroidType, shape=Shapes.N)]

    def __init__(self, date_range: TimeFrame):
        self.date_range = _validate_dates(date_range)

    @override
    def evaluate(self) -> NDArray[np.float64]:
        centroids = self.data("centroid")
        raster_vals = _make_centroid_strategy_adrio(
            "tdmean", self.date_range, centroids
        )
        return raster_vals


class Temperature(Adrio[np.float64]):
    """
    Creates an TxN matrix of floats representing the temperature in an area, represented
      in degrees Celsius (°C).
    """

    date_range: TimeFrame

    requirements = [AttributeDef("centroid", type=CentroidType, shape=Shapes.N)]

    TemperatureType = Literal["Minimum", "Mean", "Maximum"]

    temp_variables: dict[TemperatureType, str] = {
        "Minimum": "tmin",
        "Mean": "tmean",
        "Maximum": "tmax",
    }

    temp_var: TemperatureType

    def __init__(self, date_range: TimeFrame, temp_var: TemperatureType):
        self.temp_var = temp_var
        self.date_range = _validate_dates(date_range)

    @override
    def evaluate(self) -> NDArray[np.float64]:
        temp_var = self.temp_variables[self.temp_var]
        centroids = self.data("centroid")
        raster_vals = _make_centroid_strategy_adrio(
            temp_var, self.date_range, centroids
        )

        return raster_vals


class VaporPressureDeficit(Adrio[np.float64]):
    """
    Creates an TxN matrix of floats representing the vapor pressure deficit in an area,
      represented in hectopascals (hPa).
    """

    date_range: TimeFrame

    requirements = [AttributeDef("centroid", type=CentroidType, shape=Shapes.N)]

    VPDType = Literal["Minimum", "Maximum"]

    vpd_variables: dict[VPDType, str] = {"Minimum": "vpdmin", "Maximum": "vpdmax"}

    vpd_var: VPDType

    def __init__(self, date_range: TimeFrame, vpd_var: VPDType):
        self.vpd_var = vpd_var
        self.date_range = _validate_dates(date_range)

    @override
    def evaluate(self) -> NDArray[np.float64]:
        vpd_var = self.vpd_variables[self.vpd_var]
        centroids = self.data("centroid")
        raster_vals = _make_centroid_strategy_adrio(vpd_var, self.date_range, centroids)
        return raster_vals
