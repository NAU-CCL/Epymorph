"""ADRIOs thta access the US Census LODES files for commuting data."""
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing_extensions import override

from epymorph.adrio.adrio import Adrio
from epymorph.cache import load_or_fetch_url, module_cache_path
from epymorph.error import DataResourceException
from epymorph.geography.scope import GeoScope
from epymorph.geography.us_census import STATE, CensusScope, state_fips_to_code

_LODES_CACHE_PATH = module_cache_path(__name__)

# job type variables for use among all commuters classes
JobType = Literal[
    'All Jobs', 'Primary Jobs',
    'All Private Jobs', 'Private Primary Jobs',
    'All Federal Jobs', 'Federal Primary Jobs'
]

job_variables: dict[JobType, str] = {
    'All Jobs': 'JT00',
    'Primary Jobs': 'JT01',
    'All Private Jobs': 'JT02',
    'Private Primary Jobs': 'JT03',
    'All Federal Jobs': 'JT04',
    'Federal Primary Jobs': 'JT05'
}


def _fetch_lodes(scope: CensusScope, worker_type: str, job_type: str, year: int) -> NDArray[np.int64]:
    """Fetches data from LODES commuting flow data for a given year"""

    # check for valid year input
    if year not in range(2002, 2022):
        msg = "Invalid year. LODES data is only available for 2002-2021"
        raise DataResourceException(msg)

    # file type is main (residence in state only) by default
    file_type = "main"

    # initialize variables
    aggregated_data = None
    geoid = scope.get_node_ids()
    n_geocode = len(geoid)
    geocode_to_index = {geocode: i for i, geocode in enumerate(geoid)}
    geocode_len = len(geoid[0])
    data_frames = []
    # can change the lodes version, default is the most recent LODES8
    lodes_ver = "LODES8"

    if scope.granularity != 'state':
        states = STATE.truncate_list(geoid)
    else:
        states = geoid

    # check for multiple states
    if (len(states) > 1):
        file_type = "aux"

    # no federal jobs in given years
    if year in range(2002, 2010) and (job_type == "JT04" or job_type == "JT05"):

        msg = "Invalid year for job type, no federal jobs can be found between 2002 to 2009"
        raise DataResourceException(msg)

    # LODES year and state exceptions
        # exceptions can be found in this document for LODES8.1: https://lehd.ces.census.gov/data/lodes/LODES8/LODESTechDoc8.1.pdf
    invalid_conditions = [
        (year in range(2002, 2010) and (job_type == "JT04" or job_type == "JT05"),
            "Invalid year for job type, no federal jobs can be found between 2002 to 2009"),

        (('05' in states) and (year == 2002 or year in range(2019, 2022)),
            "Invalid year for state, no commuters can be found for Arkansas in 2002 or between 2019-2021"),

        (('04' in states) and (year == 2002 or year == 2003),
            "Invalid year for state, no commuters can be found for Arizona in 2002 or 2003"),

        (('11' in states) and (year in range(2002, 2010)),
            "Invalid year for state, no commuters can be found for DC in 2002 or between 2002-2009"),

        (('25' in states) and (year in range(2002, 2011)),
            "Invalid year for state, no commuters can be found for Massachusetts between 2002-2010"),

        (('28' in states) and (year in range(2002, 2004) or year in range(2019, 2022)),
            "Invalid year for state, no commuters can be found for Mississippi in 2002, 2003, or between 2019-2021"),

        (('33' in states) and year == 2002,
            "Invalid year for state, no commuters can be found for New Hampshire in 2002"),

        (('02' in states) and year in range(2017, 2022),
            "Invalid year for state, no commuters can be found for Alaska in between 2017-2021")
    ]
    for condition, message in invalid_conditions:
        if condition:
            raise DataResourceException(message)

    # translate state FIPS code to state to use in URL
    state_codes = state_fips_to_code(scope.year)
    state_abbreviations = [state_codes.get(
        fips, "").lower() for fips in states]

    for state in state_abbreviations:

        # construct the URL to fetch LODES data, reset to empty each time
        url_list = []

        # always get main file (in state residency)
        url_main = f'https://lehd.ces.census.gov/data/lodes/{lodes_ver}/{state}/od/{state}_od_main_{job_type}_{year}.csv.gz'
        url_list.append(url_main)

        # if there are more than one state in the input, get the aux files (out of state residence)
        if file_type == "aux":
            url_aux = f'https://lehd.ces.census.gov/data/lodes/{lodes_ver}/{state}/od/{state}_od_aux_{job_type}_{year}.csv.gz'
            url_list.append(url_aux)

        try:
            files = [
                load_or_fetch_url(u, _LODES_CACHE_PATH / Path(u).name)
                for u in url_list
            ]
        except Exception as e:
            raise DataResourceException("Unable to fetch LODES data.") from e

        unfiltered_df = [pd.read_csv(file, compression="gzip", converters={
            'w_geocode': str, 'h_geocode': str}) for file in files]

        # go through dataframes, multiple if there are main and aux files
        for df in unfiltered_df:

            # filter the rows on if they start with the prefix
            filtered_rows = [df[df['h_geocode'].str.startswith(
                tuple(geoid)) & df['w_geocode'].str.startswith(tuple(geoid))]]

            # add the filtered dataframe to the list of dataframes
            data_frames.append(pd.concat(filtered_rows))

    for data_df in data_frames:
        # convert w_geocode and h_geocode to strings
        data_df['w_geocode'] = data_df['w_geocode'].astype(str)
        data_df['h_geocode'] = data_df['h_geocode'].astype(str)

        # group by w_geocode and h_geocode and sum the worker values
        grouped_data = data_df.groupby(
            [data_df['w_geocode'].str[:geocode_len], data_df['h_geocode'].str[:geocode_len]])[worker_type].sum()

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
        msg = 'Census scope is required for LODES attributes.'
        raise DataResourceException(msg)

    # check if the CensusScope year is the current LODES geography: 2020
    if scope.year != 2020:
        msg = "GeoScope year does not match the LODES geography year."
        raise DataResourceException(msg)

    return scope


class Commuters(Adrio[np.int64]):
    """
    Creates an NxN matrix of integers representing the number of workers moving from a home GEOID to a work GEOID.
    """

    year: int
    """The year the data encompasses."""

    job_type: JobType

    def __init__(self, year: int, job_type: JobType = 'All Jobs'):
        self.year = year
        self.job_type = job_type

    @override
    def evaluate(self) -> NDArray[np.int64]:
        scope = self.scope
        scope = _validate_scope(scope)
        job_var = job_variables[self.job_type]
        df = _fetch_lodes(scope, "S000", job_var, self.year)
        return df


class CommutersByAge(Adrio[np.int64]):
    """
    Creates an NxN matrix of integers representing the number of workers moving from a 
    home GEOID to a work GEOID that fall under a certain age range.
    """

    year: int
    """The year the data encompasses."""

    job_type: JobType

    AgeRange = Literal[
        '29 and Under', '30_54',
        '55 and Over'
    ]

    age_variables: dict[AgeRange, str] = {
        '29 and Under': 'SA01',
        '30_54': 'SA02',
        '55 and Over': 'SA03'
    }

    age_range: AgeRange

    def __init__(self, year: int, age_range: AgeRange, job_type: JobType = 'All Jobs'):
        self.year = year
        self.age_range = age_range
        self.job_type = job_type

    @override
    def evaluate(self) -> NDArray[np.int64]:
        scope = self.scope
        scope = _validate_scope(scope)
        age_var = self.age_variables[self.age_range]
        job_var = job_variables[self.job_type]
        df = _fetch_lodes(scope, age_var, job_var, self.year)
        return df


class CommutersByEarnings(Adrio[np.int64]):
    """
    Creates an NxN matrix of integers representing the number of workers moving from a 
    home GEOID to a work GEOID that earn a certain income range monthly.
    """

    year: int
    """The year the data encompasses."""

    job_type: JobType

    EarningRange = Literal[
        '$1250 and Under', '$1251_$3333',
        '$3333 and Over'
    ]

    earnings_variables: dict[EarningRange, str] = {
        '$1250 and Under': 'SE01',
        '$1251_$3333': 'SE02',
        '$3333 and Over': 'SE03'
    }

    earning_range: EarningRange

    def __init__(self, year: int, earning_range: EarningRange, job_type: JobType = 'All Jobs'):
        self.year = year
        self.earning_range = earning_range
        self.job_type = job_type

    @override
    def evaluate(self) -> NDArray[np.int64]:
        scope = self.scope
        scope = _validate_scope(scope)
        earning_var = self.earnings_variables[self.earning_range]
        job_var = job_variables[self.job_type]
        df = _fetch_lodes(scope, earning_var, job_var, self.year)
        return df


class CommutersByIndustry(Adrio[np.int64]):
    """
    Creates an NxN matrix of integers representing the number of workers moving from a 
    home GEOID to a work GEOID that work under specified industry sector.
    """

    year: int
    """The year the data encompasses."""

    job_type: JobType

    Industries = Literal[
        'Goods Producing', 'Trade Transport Utility',
        'Other'
    ]

    industry_variables: dict[Industries, str] = {
        'Goods Producing': 'SI01',
        'Trade Transport Utility': 'SI02',
        'Other': 'SI03'
    }

    industry: Industries

    def __init__(self, year: int, industry: Industries, job_type: JobType = 'All Jobs'):
        self.year = year
        self.industry = industry
        self.job_type = job_type

    @override
    def evaluate(self) -> NDArray[np.int64]:
        scope = self.scope
        scope = _validate_scope(scope)
        industry_var = self.industry_variables[self.industry]
        job_var = job_variables[self.job_type]
        df = _fetch_lodes(scope, industry_var, job_var, self.year)
        return df
