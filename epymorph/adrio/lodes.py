"""ADRIOs that access the US Census LODES files for commuting data."""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing_extensions import override

from epymorph.adrio.adrio import Adrio, ProgressCallback
from epymorph.cache import check_file_in_cache, load_or_fetch_url, module_cache_path
from epymorph.data_usage import AvailableDataEstimate, DataEstimate
from epymorph.error import DataResourceException
from epymorph.geography.scope import GeoScope
from epymorph.geography.us_census import CensusScope
from epymorph.geography.us_geography import STATE
from epymorph.geography.us_tiger import CacheEstimate, get_states

_LODES_CACHE_PATH = module_cache_path(__name__)

_LODES_VERSION = "LODES8"
"""The current version of LODES"""

JobType = Literal[
    "All Jobs",
    "Primary Jobs",
    "All Private Jobs",
    "Private Primary Jobs",
    "All Federal Jobs",
    "Federal Primary Jobs",
]
"""Job type variables for use among all commuters classes"""

_JOB_VARIABLES: dict[JobType, str] = {
    "All Jobs": "JT00",
    "Primary Jobs": "JT01",
    "All Private Jobs": "JT02",
    "Private Primary Jobs": "JT03",
    "All Federal Jobs": "JT04",
    "Federal Primary Jobs": "JT05",
}

_STATE_FILE_ESTIMATES = {
    "ak": 970_000,
    "al": 8_300_000,
    "ar": 4_400_000,
    "az": 11_500_000,
    "ca": 72_300_000,
    "co": 10_600_000,
    "ct": 6_500_000,
    "dc": 719_000,
    "de": 1_400_000,
    "fl": 36_700_000,
    "ga": 17_900_000,
    "hi": 1_900_000,
    "ia": 6_200_000,
    "id": 2_700_000,
    "il": 26_000_000,
    "in": 12_800_000,
    "ks": 5_200_000,
    "ky": 7_200_000,
    "la": 8_100_000,
    "ma": 8_200_000,
    "md": 9_900_000,
    "me": 2_400_000,
    "mi": 18_820_000,
    "mn": 11_300_000,
    "mo": 11_300_000,
    "ms": 4_400_000,
    "mt": 1_800_000,
    "nc": 18_200_000,
    "nd": 1_400_000,
    "ne": 3_800_000,
    "nh": 2_300_000,
    "nj": 16_400_000,
    "nm": 3_100_000,
    "nv": 4_700_000,
    "ny": 35_200_000,
    "oh": 23_500_000,
    "ok": 6_700_000,
    "or": 7_300_000,
    "pa": 25_100_000,
    "ri": 1_900_000,
    "sc": 8_100_000,
    "sd": 1_500_000,
    "tn": 11_400_000,
    "tx": 50_300_000,
    "ut": 5_400_000,
    "va": 14_600_000,
    "vt": 1_080_000,
    "wa": 12_500_000,
    "wi": 11_700_000,
    "wv": 2_600_000,
    "wy": 972_000,
}
"""File estimates for JT00-JT03 main files"""


# function for creating the urls
def _generate_lodes_file_name(
    state: str, job_type: str, year: int, file_type: str
) -> tuple[str, str]:
    url_aux = ""
    url_main = f"https://lehd.ces.census.gov/data/lodes/{_LODES_VERSION}/{state}/od/{state}_od_main_{job_type}_{year}.csv.gz"

    # if there are more than one state in the input,
    # get the aux files (out of state residence)
    if file_type == "aux":
        url_aux = f"https://lehd.ces.census.gov/data/lodes/{_LODES_VERSION}/{state}/od/{state}_od_{file_type}_{job_type}_{year}.csv.gz"

    return url_aux, url_main


def _fetch_lodes(
    scope: CensusScope,
    worker_type: str,
    job_type: str,
    year: int,
    progress: ProgressCallback,
) -> NDArray[np.int64]:
    """Fetches data from LODES commuting flow data for a given year"""

    # check for valid year input
    if year not in range(2002, 2022):
        msg = "Invalid year. LODES data is only available for 2002-2021"
        raise DataResourceException(msg)

    # file type is main (residence in state only) by default
    file_type = "main"

    # initialize variables
    aggregated_data = None
    geoid = scope.node_ids
    n_geocode = len(geoid)
    geocode_to_index = {geocode: i for i, geocode in enumerate(geoid)}
    geocode_len = len(geoid[0])
    data_frames = []

    if scope.granularity != "state":
        states = list(STATE.truncate_unique(geoid))
    else:
        states = geoid

    # check for multiple states
    if len(states) > 1:
        file_type = "aux"

    # no federal jobs in given years
    if year in range(2002, 2010) and (job_type == "JT04" or job_type == "JT05"):
        msg = (
            "Invalid year for job type, no federal jobs can be found "
            "between 2002 to 2009"
        )
        raise DataResourceException(msg)

    # LODES year and state exceptions
    # exceptions can be found in this document for LODES8.1: https://lehd.ces.census.gov/data/lodes/LODES8/LODESTechDoc8.1.pdf
    invalid_conditions = [
        (
            year in range(2002, 2010) and (job_type == "JT04" or job_type == "JT05"),
            "Invalid year for job type, no federal jobs can be found "
            "between 2002 to 2009",
        ),
        (
            ("05" in states) and (year == 2002 or year in range(2019, 2022)),
            "Invalid year for state, no commuters can be found "
            "for Arkansas in 2002 or between 2019-2021",
        ),
        (
            ("04" in states) and (year == 2002 or year == 2003),
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
    for condition, message in invalid_conditions:
        if condition:
            raise DataResourceException(message)

    # translate state FIPS code to state to use in URL
    state_codes = get_states(scope.year).state_fips_to_code
    state_abbreviations = [state_codes.get(fips, "").lower() for fips in states]

    # start progress tracking
    processing_steps = len(state_abbreviations) + 1

    for i, state in enumerate(state_abbreviations):
        # construct list of urls to fetch LODES data
        url_list = []
        url_aux, url_main = _generate_lodes_file_name(state, job_type, year, file_type)

        if url_aux:
            url_list.append(url_aux)
        url_list.append(url_main)

        try:
            files = [
                load_or_fetch_url(u, _LODES_CACHE_PATH / Path(u).name) for u in url_list
            ]
        except Exception as e:
            raise DataResourceException("Unable to fetch LODES data.") from e

        # progress tracking here accounts for downloading aux and main files as one step
        # since they are being downloaded one right after the other
        if progress is not None:
            progress((i + 1) / processing_steps, None)

        unfiltered_df = [
            pd.read_csv(
                file,
                compression="gzip",
                converters={"w_geocode": str, "h_geocode": str},
            )
            for file in files
        ]

        # go through dataframes, multiple if there are main and aux files
        for df in unfiltered_df:
            # filter the rows on if they start with the prefix
            filtered_rows = [
                df[
                    df["h_geocode"].str.startswith(tuple(geoid))
                    & df["w_geocode"].str.startswith(tuple(geoid))
                ]
            ]

            # add the filtered dataframe to the list of dataframes
            data_frames.append(pd.concat(filtered_rows))

    for data_df in data_frames:
        # convert w_geocode and h_geocode to strings
        data_df["w_geocode"] = data_df["w_geocode"].astype(str)
        data_df["h_geocode"] = data_df["h_geocode"].astype(str)

        # group by w_geocode and h_geocode and sum the worker values
        grouped_data = data_df.groupby(
            [
                data_df["w_geocode"].str[:geocode_len],
                data_df["h_geocode"].str[:geocode_len],
            ]
        )[worker_type].sum()

        if aggregated_data is None:
            aggregated_data = grouped_data
        else:
            aggregated_data = aggregated_data.add(grouped_data, fill_value=0)

    # create an empty array to store worker type values
    output = np.zeros((n_geocode, n_geocode), dtype=np.int64)

    # loop through all of the grouped values and add to output
    for (w_geocode, h_geocode), value in aggregated_data.items():  # type: ignore
        w_index = geocode_to_index.get(w_geocode)
        h_index = geocode_to_index.get(h_geocode)
        output[h_index, w_index] += value

    return output


def _validate_scope(scope: GeoScope) -> CensusScope:
    if not isinstance(scope, CensusScope):
        msg = "Census scope is required for LODES attributes."
        raise DataResourceException(msg)

    # check if the CensusScope year is the current LODES geography: 2020
    if scope.year != 2020:
        msg = "GeoScope year does not match the LODES geography year."
        raise DataResourceException(msg)

    return scope


def _estimate_lodes(self, scope: CensusScope, job_type: str, year: int) -> DataEstimate:
    scope = _validate_scope(self.scope)
    est_main_size = 0
    est_aux_size = 0
    urls_aux = []
    urls_main = []
    in_cache = 0
    file_type = "main"

    # get the states to estimate data sizes
    if scope.granularity != "state":
        states = list(STATE.truncate_unique(scope.node_ids))
    else:
        states = scope.node_ids

    if len(states) > 1:
        file_type = "aux"

    # translate state FIPS code to state to use in URL
    state_codes = get_states(scope.year).state_fips_to_code
    state_abbreviations = [state_codes.get(fips, "").lower() for fips in states]

    total_state_files = len(states)

    # get the urls for each state to check the cache
    for state in state_abbreviations:
        url_aux, url_main = _generate_lodes_file_name(state, job_type, year, file_type)

        if len(states) > 1:
            urls_aux.append(url_aux)

        # if the job type is not JT04-JT05, the file size is in the dictionary
        if job_type not in ["JT04", "JT05"]:
            est_main_size += _STATE_FILE_ESTIMATES[state]
            if check_file_in_cache(_LODES_CACHE_PATH / Path(url_main).name):
                in_cache += _STATE_FILE_ESTIMATES[state]
        else:
            urls_main.append(url_main)

    # check the cache for main files if the job type is jt04-jt05
    if job_type in ["JT04", "JT05"]:
        missing_main_files = total_state_files - sum(
            1
            for u in urls_main
            if check_file_in_cache(_LODES_CACHE_PATH / Path(u).name)
        )
        # jt04-jt05 main files average to 86.2KB
        total_missing_main_size = missing_main_files * 86_200
    else:
        total_missing_main_size = (
            est_main_size - in_cache
        )  # otherwise, adjust for cache

    # check for missing aux files, if needed
    if len(states) > 1:
        # avg of 18.7KB for JT04-JT05 aux files, average of 723KB for aux otherwise
        est_aux_size = 18_700 if job_type in ["JT04", "JT05"] else 723_000
        missing_aux_files = total_state_files - sum(
            1 for u in urls_aux if check_file_in_cache(_LODES_CACHE_PATH / Path(u).name)
        )

    else:
        missing_aux_files = 0

    total_missing_aux_size = missing_aux_files * est_aux_size
    est_aux_size *= total_state_files

    est = CacheEstimate(
        total_cache_size=est_aux_size + est_main_size,
        missing_cache_size=total_missing_main_size + total_missing_aux_size,
    )

    key = f"lodes:{year}:{job_type}"
    return AvailableDataEstimate(
        name=self.full_name,
        cache_key=key,
        new_network_bytes=est.missing_cache_size,
        new_cache_bytes=est.missing_cache_size,
        total_cache_bytes=est.total_cache_size,
        max_bandwidth=None,
    )


class Commuters(Adrio[np.int64]):
    """
    Creates an NxN matrix of integers representing the number of workers moving
    from a home GEOID to a work GEOID.
    """

    year: int
    """The year the data encompasses."""

    job_type: JobType

    def __init__(self, year: int, job_type: JobType = "All Jobs"):
        """
        Initializes the commuters matrix with the year and with the type of job, which
        defaults to "All Jobs".

        Parameters
        ----------
        year : int
            The year for all of the commuting data.
        job_type : JobType
            The category of job all resulting commuters work under (options: "All Jobs",
            "Primary Jobs", "All Private Jobs", "Private Primary Jobs",
            "All Federal Jobs", "Federal Primary Jobs").
        """
        self.year = year
        self.job_type = job_type

    def estimate_data(self) -> DataEstimate:
        scope = self.scope
        scope = _validate_scope(scope)
        job_var = _JOB_VARIABLES[self.job_type]
        est = _estimate_lodes(self, scope, job_var, self.year)
        return est

    @override
    def evaluate_adrio(self) -> NDArray[np.int64]:
        scope = self.scope
        scope = _validate_scope(scope)
        job_var = _JOB_VARIABLES[self.job_type]
        return _fetch_lodes(scope, "S000", job_var, self.year, self.progress)


class CommutersByAge(Adrio[np.int64]):
    """
    Creates an NxN matrix of integers representing the number of workers moving from a
    home GEOID to a work GEOID that fall under a certain age range.
    """

    year: int
    """The year the data encompasses."""

    job_type: JobType

    AgeRange = Literal["29 and Under", "30_54", "55 and Over"]

    age_variables: dict[AgeRange, str] = {
        "29 and Under": "SA01",
        "30_54": "SA02",
        "55 and Over": "SA03",
    }

    age_range: AgeRange

    def __init__(self, year: int, age_range: AgeRange, job_type: JobType = "All Jobs"):
        """
        Initializes the commuters by age matrix with the year, the age range, and the
        type of job, which defaults to "All Jobs".

        Parameters
        ----------
        year : int
            The year for all of the commuting data.
        age_range : AgeRange
            The defined age range that all commuters fall under
            (options: '29 and Under', '30_54', '55 and Over').
        job_type : JobType
            The category of job all resulting commuters work under (options: "All Jobs",
            "Primary Jobs", "All Private Jobs", "Private Primary Jobs",
            "All Federal Jobs", "Federal Primary Jobs").
        """
        self.year = year
        self.age_range = age_range
        self.job_type = job_type

    def estimate_data(self) -> DataEstimate:
        scope = self.scope
        scope = _validate_scope(scope)
        job_var = _JOB_VARIABLES[self.job_type]
        est = _estimate_lodes(self, scope, job_var, self.year)
        return est

    @override
    def evaluate_adrio(self) -> NDArray[np.int64]:
        scope = self.scope
        scope = _validate_scope(scope)
        age_var = self.age_variables[self.age_range]
        job_var = _JOB_VARIABLES[self.job_type]
        return _fetch_lodes(scope, age_var, job_var, self.year, self.progress)


class CommutersByEarnings(Adrio[np.int64]):
    """
    Creates an NxN matrix of integers representing the number of workers moving from a
    home GEOID to a work GEOID that earn a certain income range monthly.
    """

    year: int
    """The year the data encompasses."""

    job_type: JobType

    EarningRange = Literal["$1250 and Under", "$1251_$3333", "$3333 and Over"]

    earnings_variables: dict[EarningRange, str] = {
        "$1250 and Under": "SE01",
        "$1251_$3333": "SE02",
        "$3333 and Over": "SE03",
    }

    earning_range: EarningRange

    def __init__(
        self, year: int, earning_range: EarningRange, job_type: JobType = "All Jobs"
    ):
        """
        Initializes the commuters by monthly earnings matrix with the year, the earning
        range, and the type of job, which defaults to "All Jobs".

        Parameters
        ----------
        year : int
            The year for all of the commuting data.
        earning_range : EarningRange
            The defined monthly earnings range that all commuters fall under
            (options: '$1250 and Under', '$1251_$3333', '$3333 and Over').
        job_type : JobType
            The category of job all resulting commuters work under (options: "All Jobs",
            "Primary Jobs", "All Private Jobs", "Private Primary Jobs",
            "All Federal Jobs", "Federal Primary Jobs").
        """
        self.year = year
        self.earning_range = earning_range
        self.job_type = job_type

    def estimate_data(self) -> DataEstimate:
        scope = self.scope
        scope = _validate_scope(scope)
        job_var = _JOB_VARIABLES[self.job_type]
        est = _estimate_lodes(self, scope, job_var, self.year)
        return est

    @override
    def evaluate_adrio(self) -> NDArray[np.int64]:
        scope = self.scope
        scope = _validate_scope(scope)
        earning_var = self.earnings_variables[self.earning_range]
        job_var = _JOB_VARIABLES[self.job_type]
        return _fetch_lodes(scope, earning_var, job_var, self.year, self.progress)


class CommutersByIndustry(Adrio[np.int64]):
    """
    Creates an NxN matrix of integers representing the number of workers moving from a
    home GEOID to a work GEOID that work under specified industry sector.
    """

    year: int
    """The year the data encompasses."""

    job_type: JobType

    Industries = Literal["Goods Producing", "Trade Transport Utility", "Other"]

    industry_variables: dict[Industries, str] = {
        "Goods Producing": "SI01",
        "Trade Transport Utility": "SI02",
        "Other": "SI03",
    }

    industry: Industries

    def __init__(self, year: int, industry: Industries, job_type: JobType = "All Jobs"):
        """
        Initializes the commuters by industry matrix with the year, the industry type,
        and the type of job, which defaults to "All Jobs".

        Parameters
        ----------
        year : int
            The year for all of the commuting data.
        industry : Industries
            The defined industry that all commuters fall under
            (options: 'Goods Producing', 'Trade Transport Utility', 'Other').
        job_type : JobType
            The category of job all resulting commuters work under (options: "All Jobs",
            "Primary Jobs", "All Private Jobs", "Private Primary Jobs",
            "All Federal Jobs", "Federal Primary Jobs").
        """
        self.year = year
        self.industry = industry
        self.job_type = job_type

    def estimate_data(self) -> DataEstimate:
        scope = self.scope
        scope = _validate_scope(scope)
        job_var = _JOB_VARIABLES[self.job_type]
        est = _estimate_lodes(self, scope, job_var, self.year)
        return est

    @override
    def evaluate_adrio(self) -> NDArray[np.int64]:
        scope = self.scope
        scope = _validate_scope(scope)
        industry_var = self.industry_variables[self.industry]
        job_var = _JOB_VARIABLES[self.job_type]
        return _fetch_lodes(scope, industry_var, job_var, self.year, self.progress)
