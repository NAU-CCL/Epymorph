"""ADRIOs that access PRISM files for climate data."""

from abc import abstractmethod
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Literal, NamedTuple, Protocol

import numpy as np
import rasterio.io as rio
from dateutil.relativedelta import relativedelta
from numpy.typing import NDArray
from typing_extensions import override

from epymorph.adrio.adrio import (
    ADRIO,
    ADRIOCommunicationError,
    ADRIOContextError,
    InspectResult,
    adrio_cache,
)
from epymorph.adrio.validation import (
    ResultFormat,
)
from epymorph.attribute import AttributeDef
from epymorph.cache import check_file_in_cache, load_or_fetch_url, module_cache_path
from epymorph.data_shape import Shapes
from epymorph.data_type import CentroidType
from epymorph.data_usage import AvailableDataEstimate, DataEstimate
from epymorph.geography.us_census import CensusScope
from epymorph.geography.us_geography import STATE
from epymorph.simulation import Context
from epymorph.time import TimeFrame

_PRISM_CACHE_PATH = module_cache_path(__name__)

TemperatureType = Literal["Minimum", "Mean", "Maximum"]
"""A daily temperature measurement provided in the PRISM data."""

VPDType = Literal["Minimum", "Maximum"]
"""A daily vapor pressure deficit measurement provided in the PRISM data."""


class _PrismFile(NamedTuple):
    url: str
    bil_name: str
    cache_path: Path


def _prism_files(time_frame: TimeFrame, attribute: str) -> list[_PrismFile]:
    """Compute the complete list of PRISM files needed."""
    latest_date = date.today() - timedelta(days=1)
    six_months_ago = date.today() + relativedelta(months=-6)
    last_completed_month = six_months_ago.replace(day=1) - timedelta(days=1)

    files = []
    for file_date in time_frame:
        if file_date.year == latest_date.year and file_date.month == latest_date.month:
            stability = "early"
        elif file_date > last_completed_month:
            # if it is before the last finished month
            stability = "provisional"
        else:
            # if it is older than 6 completed months
            stability = "stable"

        name = f"PRISM_{attribute}_{stability}_4kmD2_{file_date.strftime('%Y%m%d')}_bil"
        files.append(
            _PrismFile(
                url=f"https://ftp.prism.oregonstate.edu/daily/{attribute}/{file_date.year}/{name}.zip",
                bil_name=f"{name}.bil",
                cache_path=_PRISM_CACHE_PATH / f"{name}.zip",
            )
        )
    return files


class _Sampler(Protocol):
    """A strategy for sampling values from the rasterized PRISM files."""

    # NOTE: although currently there's only one implementation of _Sampler,
    # in future, it may be possible to choose or provide a sampler to the PRISM ADRIOs.

    @abstractmethod
    def sample(self, raster: rio.DatasetReader, centroids: NDArray) -> list[float]:
        pass


class _CentroidSampler(_Sampler):
    """A sampler that uses the data value of the pixel containing each centroid."""

    @override
    def sample(self, raster: rio.DatasetReader, centroids: NDArray) -> list[float]:
        return [x[0] for x in raster.sample(centroids)]


class _PrismADRIOMixin(ADRIO[np.float64, np.float64]):
    """Shared functionality for PRISM ADRIOs."""

    _CENTROID = AttributeDef("centroid", type=CentroidType, shape=Shapes.N)

    requirements = (_CENTROID,)

    @property
    @override
    def result_format(self) -> ResultFormat:
        return ResultFormat(shape=Shapes.TxN, dtype=np.float64)

    @override
    def validate_context(self, context: Context) -> None:
        scope = self.scope
        if isinstance(scope, CensusScope):
            # Hawaii, Alaska, and Puerto Rico aren't included in the standard dataset
            states = list(STATE.truncate_unique(scope.node_ids))
            if any(x in ("72", "02", "15") for x in states):
                err = (
                    "Alaska, Hawaii, and Puerto Rico cannot be evaluated for PRISM "
                    "attributes. Please use geo scopes within the 48 contiguous states."
                )
                raise ADRIOContextError(self, context, err)

        # PRISM only accounts for dates from 1981 up to yesterday's date
        earliest_date = date(1981, 1, 1)
        latest_date = date.today() - timedelta(days=1)
        tf = self.time_frame
        if tf.start_date < earliest_date or tf.end_date > latest_date:
            err = (
                "Given date range is out of PRISM scope, please enter dates between "
                f"1981-01-01 and {latest_date}"
            )
            raise ADRIOContextError(self, context, err)

    @property
    @abstractmethod
    def _attribute_name(self) -> str:
        pass

    @property
    @abstractmethod
    def _file_size(self) -> int:
        pass

    @override
    def estimate_data(self) -> DataEstimate:
        time_frame = self.time_frame
        attribute = self._attribute_name

        files = _prism_files(time_frame, attribute)
        total_files = len(files)
        missing_files = sum(1 for _, _, path in files if not check_file_in_cache(path))

        est_file_size = self._file_size
        total = total_files * est_file_size
        missing = missing_files * est_file_size

        return AvailableDataEstimate(
            name=self.class_name,
            cache_key=f"prism:{attribute}:{time_frame}",
            new_network_bytes=missing,
            new_cache_bytes=missing,
            total_cache_bytes=total,
            max_bandwidth=None,
        )

    def _fetch_all(self, files: Iterable[_PrismFile]) -> Iterable[rio.DatasetReader]:
        for url, bil_name, cache_path in files:
            try:
                file = load_or_fetch_url(url, cache_path)
                with rio.ZipMemoryFile(file) as zip_contents:
                    with zip_contents.open(bil_name) as dataset:
                        yield dataset
            except Exception as e:
                err = "Unable to fetch PRISM data."
                raise ADRIOCommunicationError(self, self.context, err) from e

    @override
    def inspect(self) -> InspectResult[np.float64, np.float64]:
        self.validate_context(self.context)

        centroids = self.data(self._CENTROID)
        attribute = self._attribute_name
        files = _prism_files(self.time_frame, attribute)

        sampler = _CentroidSampler()

        processing_steps = len(files) + 1
        daily_values = []
        for i, raster in enumerate(self._fetch_all(files)):
            vals = sampler.sample(raster, centroids)
            daily_values.append(vals)
            self._report_progress((i + 1) / processing_steps)
        result_np = np.array(daily_values, dtype=np.float64)

        # sentinel value -9999 means undefined
        issues = {}
        undefined_mask = np.isclose(result_np, -9999.0)
        if np.any(undefined_mask):
            issues["undefined"] = undefined_mask
            result_np = np.ma.masked_array(result_np, mask=undefined_mask)

        self.validate_result(self.context, result_np)
        return InspectResult(
            adrio=self,
            source=None,  # source data isn't suitable to share
            result=result_np,
            dtype=self.result_format.dtype.type,
            shape=self.result_format.shape,
            issues=issues,
        )


@adrio_cache
class Precipitation(_PrismADRIOMixin, ADRIO[np.float64, np.float64]):
    """
    Loads daily precipitation data (in millimeters) from PRISM data files.

    PRISM ADRIOs require you to provide centroids as a data attribute, which is where
    we'll sample a value from the source file for each node. If the PRISM model is not
    defined for a particular centroid, you'll get an "undefined" data issue in the
    result. This is common for locations outside of the US, as PRISM only covers the
    contiguous 48 United States; but it also happens for centroids located on bodies of
    water. In this case, you may need to adjust centroids to get values.
    """

    @property
    @override
    def _attribute_name(self) -> str:
        return "ppt"

    @property
    @override
    def _file_size(self) -> int:
        return 1_200_000


@adrio_cache
class DewPoint(_PrismADRIOMixin, ADRIO[np.float64, np.float64]):
    """
    Loads daily dew point data (in degrees Celsius) from PRISM data files.

    PRISM ADRIOs require you to provide centroids as a data attribute, which is where
    we'll sample a value from the source file for each node. If the PRISM model is not
    defined for a particular centroid, you'll get an "undefined" data issue in the
    result. This is common for locations outside of the US, as PRISM only covers the
    contiguous 48 United States; but it also happens for centroids located on bodies of
    water. In this case, you may need to adjust centroids to get values.
    """

    @property
    @override
    def _attribute_name(self) -> str:
        return "tdmean"

    @property
    @override
    def _file_size(self) -> int:
        if self.time_frame.start_date.year > 2020:
            return 1_800_000  # average to 1.8MB after 2020
        else:
            return 1_400_000  # average to 1.4MB 2020 and before


@adrio_cache
class Temperature(_PrismADRIOMixin, ADRIO[np.float64, np.float64]):
    """
    Loads daily temperature data (in degrees Celsius) from PRISM data files.
    You can select either the minimum, maximum, or mean daily temperature.

    PRISM ADRIOs require you to provide centroids as a data attribute, which is where
    we'll sample a value from the source file for each node. If the PRISM model is not
    defined for a particular centroid, you'll get an "undefined" data issue in the
    result. This is common for locations outside of the US, as PRISM only covers the
    contiguous 48 United States; but it also happens for centroids located on bodies of
    water. In this case, you may need to adjust centroids to get values.

    Parameters
    ----------
    temp_var :
        Which daily temperature measure to select (min, max, mean).
    """

    _temp_var: TemperatureType

    def __init__(self, temp_var: TemperatureType):
        self._temp_var = temp_var

    @property
    @override
    def _attribute_name(self) -> str:
        match self._temp_var:
            case "Minimum":
                return "tmin"
            case "Mean":
                return "tmean"
            case "Maximum":
                return "tmax"

    @property
    @override
    def _file_size(self) -> int:
        if self.time_frame.start_date.year > 2020:
            return 1_700_000  # average to 1.7MB after 2020
        else:
            return 1_400_000  # average to 1.4MB 2020 and before


@adrio_cache
class VaporPressureDeficit(_PrismADRIOMixin, ADRIO[np.float64, np.float64]):
    """
    Loads daily vapor pressure deficit data (in hectopascals) from PRISM data files.
    You can select either the minimum or maximum daily vapor pressure deficit.

    PRISM ADRIOs require you to provide centroids as a data attribute, which is where
    we'll sample a value from the source file for each node. If the PRISM model is not
    defined for a particular centroid, you'll get an "undefined" data issue in the
    result. This is common for locations outside of the US, as PRISM only covers the
    contiguous 48 United States; but it also happens for centroids located on bodies of
    water. In this case, you may need to adjust centroids to get values.

    Parameters
    ----------
    vpd_var :
        Which daily vapor pressure deficit measure to select (min, max).
    """

    _vpd_var: VPDType

    def __init__(self, vpd_var: VPDType):
        self._vpd_var = vpd_var

    @property
    @override
    def _attribute_name(self) -> str:
        match self._vpd_var:
            case "Minimum":
                return "vpdmin"
            case "Maximum":
                return "vpdmax"

    @property
    @override
    def _file_size(self) -> int:
        if self.time_frame.start_date.year > 2020:
            return 1_700_000  # average to 1.7MB after 2020
        else:
            return 1_300_000  # average to 1.3MB 2020 and before
