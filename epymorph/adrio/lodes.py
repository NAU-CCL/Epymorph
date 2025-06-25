"""ADRIOs that access the US Census LODES files for commuting data."""

from abc import abstractmethod
from pathlib import Path
from typing import Literal, NamedTuple, cast

import numpy as np
import pandas as pd
from typing_extensions import override

from epymorph.adrio.adrio import (
    ADRIO,
    ADRIOCommunicationError,
    ADRIOContextError,
    InspectResult,
    adrio_cache,
)
from epymorph.adrio.validation import ResultFormat
from epymorph.cache import check_file_in_cache, load_or_fetch_url, module_cache_path
from epymorph.data_shape import Shapes
from epymorph.data_usage import AvailableDataEstimate, DataEstimate
from epymorph.error import MissingContextError
from epymorph.geography.us_census import CensusScope
from epymorph.geography.us_geography import STATE, CensusGranularity
from epymorph.geography.us_tiger import get_states
from epymorph.simulation import Context

_LODES_CACHE_PATH = module_cache_path(__name__)

JobType = Literal[
    "All Jobs",
    "Primary Jobs",
    "All Private Jobs",
    "Private Primary Jobs",
    "All Federal Jobs",
    "Federal Primary Jobs",
]
"""A job type that LODES provides subtotals for."""

AgeRange = Literal["29 and Under", "30_54", "55 and Over"]
"""Age ranges that LODES provides subtotals for."""

EarningRange = Literal["$1250 and Under", "$1251_$3333", "$3333 and Over"]
"""Earning ranges that LODES provides subtotals for."""

Industry = Literal["Goods Producing", "Trade Transport Utility", "Other"]
"""Job industries that LODES provides subtotals for."""

_JobCode = Literal["JT00", "JT01", "JT02", "JT03", "JT04", "JT05"]
_FileType = Literal["main", "aux"]

# fmt: off
_STATE_FILE_ESTIMATES = {
    "ak": 970_000,    "al": 8_300_000,  "ar": 4_400_000,  "az": 11_500_000,
    "ca": 72_300_000, "co": 10_600_000, "ct": 6_500_000,  "dc": 719_000,
    "de": 1_400_000,  "fl": 36_700_000, "ga": 17_900_000, "hi": 1_900_000,
    "ia": 6_200_000,  "id": 2_700_000,  "il": 26_000_000, "in": 12_800_000,
    "ks": 5_200_000,  "ky": 7_200_000,  "la": 8_100_000,  "ma": 8_200_000,
    "md": 9_900_000,  "me": 2_400_000,  "mi": 18_820_000, "mn": 11_300_000,
    "mo": 11_300_000, "ms": 4_400_000,  "mt": 1_800_000,  "nc": 18_200_000,
    "nd": 1_400_000,  "ne": 3_800_000,  "nh": 2_300_000,  "nj": 16_400_000,
    "nm": 3_100_000,  "nv": 4_700_000,  "ny": 35_200_000, "oh": 23_500_000,
    "ok": 6_700_000,  "or": 7_300_000,  "pa": 25_100_000, "ri": 1_900_000,
    "sc": 8_100_000,  "sd": 1_500_000,  "tn": 11_400_000, "tx": 50_300_000,
    "ut": 5_400_000,  "va": 14_600_000, "vt": 1_080_000,  "wa": 12_500_000,
    "wi": 11_700_000, "wv": 2_600_000,  "wy": 972_000,
}
"""File estimates for JT00-JT03 main files"""
# fmt: on


def _file_size(state: str, file: _FileType, job: _JobCode) -> int:
    """Estimate the file size in bytes of a LODES file."""
    match (file, job):
        case ("main", "JT04" | "JT05"):
            return 86_200  # 86.2KB
        case ("main", _):
            return _STATE_FILE_ESTIMATES[state]
        case ("aux", "JT04" | "JT05"):
            return 18_700  # 18.7KB
        case ("aux", _):
            return 723_000  # 723KB


class _LodesFile(NamedTuple):
    state: str
    file: _FileType
    url: str
    cache_path: Path


def _lodes_files(scope: CensusScope, year: int, job: _JobCode) -> list[_LodesFile]:
    """Compute the complete list of LODES files needed."""
    version = "LODES8"  # we only support LODES v8 currently

    def file_for(state, file_type):
        url = f"https://lehd.ces.census.gov/data/lodes/{version}/{state}/od/{state}_od_{file_type}_{job}_{year}.csv.gz"
        cache_path = _LODES_CACHE_PATH / Path(url).name
        return _LodesFile(state, file_type, url, cache_path)

    # files are state-based
    states = list(STATE.truncate_unique(scope.node_ids))
    state_fips_to_code = get_states(scope.year).state_fips_to_code
    state_codes = [state_fips_to_code[x].lower() for x in states]

    # if there's more than one state, we will need to load both main and aux files
    file_types: list[_FileType] = ["main", "aux"] if len(states) > 1 else ["main"]

    return [file_for(s, f) for s in state_codes for f in file_types]


class _LodesADRIOMixin(ADRIO[np.int64, np.int64]):
    """Shared functionality for LODES ADRIOs."""

    _override_year: int | None
    """Selected data year."""
    _job_type: JobType
    """Selected job type."""

    @property
    def _job_code(self) -> _JobCode:
        """The job code corresponding to the selected job type."""
        match self._job_type:
            case "All Jobs":
                return "JT00"
            case "Primary Jobs":
                return "JT01"
            case "All Private Jobs":
                return "JT02"
            case "Private Primary Jobs":
                return "JT03"
            case "All Federal Jobs":
                return "JT04"
            case "Federal Primary Jobs":
                return "JT05"

    @property
    @abstractmethod
    def _worker_type(self) -> str:
        """The worker type code identifying the data column to use from LODES files."""

    @property
    def data_year(self) -> int:
        """
        The data year to fetch. If this ADRIO was constructed with the `year` argument,
        that is used. Otherwise default to the year in which the simulation time frame
        starts.
        """
        if self._override_year is not None:
            return self._override_year
        return self.time_frame.start_date.year

    @property
    @override
    def result_format(self) -> ResultFormat:
        return ResultFormat(dtype=np.int64, shape=Shapes.NxN)

    @override
    def validate_context(self, context: Context) -> None:
        try:
            scope = self.scope
        except MissingContextError:
            err = "Census scope is required for LODES attributes."
            raise ADRIOContextError(self, context, err)
        if not isinstance(scope, CensusScope):
            err = "Census scope is required for LODES attributes."
            raise ADRIOContextError(self, context, err)
        if scope.year != 2020:
            err = "Invalid scope year; LODES requires 2020 Census geography."
            raise ADRIOContextError(self, context, err)

        year = self.data_year
        if year not in range(2002, 2022):
            err = "Invalid year. LODES data is only available for 2002-2021"
            raise ValueError(err)

        # LODES year and state exceptions
        # exceptions can be found in this document for LODES8.1: https://lehd.ces.census.gov/data/lodes/LODES8/LODESTechDoc8.1.pdf
        states = list(STATE.truncate_unique(scope.node_ids))
        invalid_conditions = [
            (
                (year in range(2002, 2010)) and self._job_code in ("JT04", "JT05"),
                "Federal job commuters data is not available between 2002 and 2009.",
            ),
            (
                ("05" in states) and (year == 2002 or year in range(2019, 2022)),
                "Invalid year for state, no commuters can be found "
                "for Arkansas in 2002 or between 2019-2021",
            ),
            (
                ("04" in states) and (year in (2002, 2003)),
                "Invalid year for state, no commuters can be found "
                "for Arizona in 2002 or 2003",
            ),
            (
                ("11" in states) and (year in range(2002, 2010)),
                "Invalid year for state, no commuters can be found "
                "for DC in 2002 or between 2002-2009",
            ),
            (
                ("25" in states) and (year in range(2002, 2011)),
                "Invalid year for state, no commuters can be found "
                "for Massachusetts between 2002-2010",
            ),
            (
                ("28" in states)
                and (year in range(2002, 2004) or year in range(2019, 2022)),
                "Invalid year for state, no commuters can be found "
                "for Mississippi in 2002, 2003, or between 2019-2021",
            ),
            (
                ("33" in states) and year == 2002,
                "Invalid year for state, no commuters can be found "
                "for New Hampshire in 2002",
            ),
            (
                ("02" in states) and year in range(2017, 2022),
                "Invalid year for state, no commuters can be found "
                "for Alaska in between 2017-2021",
            ),
        ]
        for condition, err in invalid_conditions:
            if condition:
                raise ADRIOContextError(self, context, err)

    @override
    def estimate_data(self) -> DataEstimate:
        scope = cast(CensusScope, self.scope)
        job_code = self._job_code
        year = self.data_year

        # for each file, a tuple: (size estimate, already cached?)
        files = [
            (_file_size(state, file, job_code), check_file_in_cache(cache_path))
            for state, file, _, cache_path in _lodes_files(scope, year, job_code)
        ]

        total = sum(size for size, _ in files)
        missing = sum(size for size, in_cache in files if not in_cache)

        return AvailableDataEstimate(
            name=self.class_name,
            cache_key=f"lodes:{year}:{job_code}",
            new_network_bytes=missing,
            new_cache_bytes=missing,
            total_cache_bytes=total,
            max_bandwidth=None,
        )

    @override
    def inspect(self) -> InspectResult[np.int64, np.int64]:
        self.validate_context(self.context)
        scope = cast(CensusScope, self.scope)
        node_ids: tuple[str, ...] = tuple(scope.node_ids)

        worker_type = self._worker_type

        # load the data from each LODES file
        data_frames = []
        files = _lodes_files(scope, self.data_year, self._job_code)
        processing_steps = len(files) + 1
        for i, file in enumerate(files):
            try:
                file_df = pd.read_csv(
                    load_or_fetch_url(file.url, file.cache_path),
                    compression="gzip",
                    dtype={"w_geocode": str, "h_geocode": str, worker_type: int},
                    usecols=["w_geocode", "h_geocode", worker_type],
                )

                # filter to locations that we're interested in
                file_df = file_df[
                    file_df["h_geocode"].str.startswith(node_ids)
                    & file_df["w_geocode"].str.startswith(node_ids)
                ]
                data_frames.append(file_df)

                self._report_progress((i + 1) / processing_steps)
            except Exception as e:
                err = "Unable to fetch LODES data."
                raise ADRIOCommunicationError(self, self.context, err) from e

        # Combine LODES data, group by the nodes in our scope, and pivot/sum
        full_df = pd.concat(data_frames)

        geoid_len = CensusGranularity.of(scope.granularity).length

        result_np = (
            full_df.assign(
                w_geocode=full_df["w_geocode"].str[:geoid_len],
                h_geocode=full_df["h_geocode"].str[:geoid_len],
            )
            .pivot_table(
                index="h_geocode",
                columns="w_geocode",
                values=worker_type,
                aggfunc="sum",
                fill_value=0,
                sort=True,
            )
            .reindex(
                index=scope.node_ids,
                columns=scope.node_ids,
                fill_value=0,
            )
            .to_numpy(dtype=np.int64)
        )

        self.validate_result(self.context, result_np)
        return InspectResult(
            adrio=self,
            source=full_df,
            result=result_np,
            dtype=self.result_format.dtype.type,
            shape=self.result_format.shape,
            issues={},
        )


@adrio_cache
class Commuters(_LodesADRIOMixin, ADRIO[np.int64, np.int64]):
    """
    Loads data from the US Census Bureau's Longitudinal Employer-Household Dynamics
    Origin-Destination Employment Statistics (LODES) data product, version 8.
    LODES provides counts of individuals living in one location and working in another,
    and various subtotals are available.

    The product aggregates to 2020 census blocks, so this ADRIO can work with scopes
    from state-granularity down to block-group-granularity. The geography year must be
    2020, however the data itself is computed yearly from 2002 through 2022.

    The result is an NxN matrix of integers, with residency location on the first axis
    and work location on the second axis.

    Parameters
    ----------
    year :
        The year for the commuting data.
        Defaults to the year in which the simulation time frame starts.
    job_type :
        The job category used to filter commuters.
    """

    def __init__(self, *, year: int | None = None, job_type: JobType = "All Jobs"):
        self._override_year = year
        self._job_type = job_type

    @property
    @override
    def _worker_type(self) -> str:
        return "S000"


@adrio_cache
class CommutersByAge(_LodesADRIOMixin, ADRIO[np.int64, np.int64]):
    """
    Loads data from the US Census Bureau's Longitudinal Employer-Household Dynamics
    Origin-Destination Employment Statistics (LODES) data product, version 8.
    LODES provides counts of individuals living in one location and working in another,
    and various subtotals are available.

    The product aggregates to 2020 census blocks, so this ADRIO can work with scopes
    from state-granularity down to block-group-granularity. The geography year must be
    2020, however the data itself is computed yearly from 2002 through 2022.

    This ADRIO filters by age groups of the workers.
    The result is an NxN matrix of integers, with residency location on the first axis
    and work location on the second axis.

    Parameters
    ----------
    age_range :
        The age range used to filter commuters.
    year :
        The year for the commuting data.
        Defaults to the year in which the simulation time frame starts.
    job_type :
        The job category used to filter commuters.
    """

    _age_range: AgeRange

    def __init__(
        self,
        age_range: AgeRange,
        *,
        year: int | None = None,
        job_type: JobType = "All Jobs",
    ):
        self._override_year = year
        self._job_type = job_type
        self._age_range = age_range

    @property
    @override
    def _worker_type(self) -> str:
        match self._age_range:
            case "29 and Under":
                return "SA01"
            case "30_54":
                return "SA02"
            case "55 and Over":
                return "SA03"


@adrio_cache
class CommutersByEarnings(_LodesADRIOMixin, ADRIO[np.int64, np.int64]):
    """
    Loads data from the US Census Bureau's Longitudinal Employer-Household Dynamics
    Origin-Destination Employment Statistics (LODES) data product, version 8.
    LODES provides counts of individuals living in one location and working in another,
    and various subtotals are available.

    The product aggregates to 2020 census blocks, so this ADRIO can work with scopes
    from state-granularity down to block-group-granularity. The geography year must be
    2020, however the data itself is computed yearly from 2002 through 2022.

    This ADRIO filters by the monthly earning bracket of the workers.
    The result is an NxN matrix of integers, with residency location on the first axis
    and work location on the second axis.

    Parameters
    ----------
    earning_range :
        The monthly earnings range used to filter commuters.
    year :
        The year for the commuting data.
        Defaults to the year in which the simulation time frame starts.
    job_type :
        The job category used to filter commuters.
    """

    _earning_range: EarningRange
    """The monthly earnings range used to filter commuters."""

    def __init__(
        self,
        earning_range: EarningRange,
        *,
        year: int | None = None,
        job_type: JobType = "All Jobs",
    ):
        self._override_year = year
        self._job_type = job_type
        self._earning_range = earning_range

    @property
    @override
    def _worker_type(self) -> str:
        match self._earning_range:
            case "$1250 and Under":
                return "SE01"
            case "$1251_$3333":
                return "SE02"
            case "$3333 and Over":
                return "SE03"


@adrio_cache
class CommutersByIndustry(_LodesADRIOMixin, ADRIO[np.int64, np.int64]):
    """
    Loads data from the US Census Bureau's Longitudinal Employer-Household Dynamics
    Origin-Destination Employment Statistics (LODES) data product, version 8.
    LODES provides counts of individuals living in one location and working in another,
    and various subtotals are available.

    The product aggregates to 2020 census blocks, so this ADRIO can work with scopes
    from state-granularity down to block-group-granularity. The geography year must be
    2020, however the data itself is computed yearly from 2002 through 2022.

    This ADRIO filters by the job industry of the workers.
    The result is an NxN matrix of integers, with residency location on the first axis
    and work location on the second axis.

    Parameters
    ----------
    industry :
        The industry used to filter commuters.
    year :
        The year for the commuting data.
        Defaults to the year in which the simulation time frame starts.
    job_type :
        The job category used to filter commuters.
    """

    _industry: Industry
    """The industry used to filter commuters."""

    def __init__(
        self,
        industry: Industry,
        *,
        year: int | None = None,
        job_type: JobType = "All Jobs",
    ):
        self._override_year = year
        self._job_type = job_type
        self._industry = industry

    @property
    @override
    def _worker_type(self) -> str:
        match self._industry:
            case "Goods Producing":
                return "SI01"
            case "Trade Transport Utility":
                return "SI02"
            case "Other":
                return "SI03"
