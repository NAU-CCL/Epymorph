from copy import copy
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path
from turtle import home
from urllib.request import urlopen
from warnings import warn

import numpy as np
import pandas as pd
import plot_utils as pu
from numpy.typing import NDArray
from pandas import DataFrame, concat, read_excel
from pygris import tracts
from pygris.data import get_lodes

from epymorph.cache import (CacheMiss, CacheWarning, load_file_from_cache,
                            save_file_to_cache)
from epymorph.data_shape import Shapes
from epymorph.error import GeoValidationException
from epymorph.geo.adrio.adrio import ADRIO, ADRIOMaker
from epymorph.geo.spec import Geography, TimePeriod, Year
from epymorph.geography.us_census import (BLOCK, BLOCK_GROUP,
                                          CENSUS_GRANULARITY, COUNTY, STATE,
                                          TRACT, state_fips_to_code)
from epymorph.geography.us_tiger import _fetch_url
from epymorph.simulation import AttributeDef, geo_attrib

_LODES_CACHE_PATH = Path("geo/adrio/census/lodes")


class Granularity(Enum):
    """Enumeration of LODES granularity levels."""
    STATE = "state"
    COUNTY = "county"
    TRACT = "tract"
    CBG = "block group"
    BLOCK = "block"


@dataclass
class LodesGeography(Geography):
    """Dataclass used to describe geographies for LODES ADRIOs/"""

    granularity: Granularity
    """The granularity level to fetch data from"""

    filter: dict[str, list[str]]
    """A list of locations to fetch data from as FIPS codes for relevant granularities"""


class ADRIOMakerLODES(ADRIOMaker):
    """
    LODES8 ADRIO template to serve as parent class and provide utility functions for LODES8-based ADRIOS.
    """

    # origin-destination files query
    employment_query = ["S000",  # all total jobs
                        "SA01",  # jobs by age (29 or under, 30-54, 55+)
                        "SA02",
                        "SA03",
                        # jobs by monthly earnings ($1250 or under, $1251-$3333, $3333+ )
                        "SE01",
                        "SE02",
                        "SE03",
                        # jobs by industry (Goods Producing, Trade+Transportation+Utilities, All others)
                        "SI01",
                        "SI02",
                        "SI03"]

    attributes = [
        geo_attrib('name', dtype=str, shape=Shapes.N,
                   comment='The proper name of the place.'),
        geo_attrib('commuters', dtype=int, shape=Shapes.NxN,
                   comment='The number of total commuters from the work geoid to the home geoid'),
        geo_attrib('commuters_29_under', dtype=int, shape=Shapes.NxN,
                   comment='The number of total commuters ages 29 and under from the work geoid to the home geoid'),
        geo_attrib('commuters_30_to_54', dtype=int, shape=Shapes.NxN,
                   comment='The number of total commuters between ages 30 to 54 from the work geoid to the home geoid'),
        geo_attrib('commuters_55_over', dtype=int, shape=Shapes.NxN,
                   comment='The number of total commuters ages 55 and over from the work geoid to the home geoid'),
        geo_attrib('commuters_1250_under_earnings', dtype=int, shape=Shapes.NxN,
                   comment='The number of total commuters that earn $1250 or under monthly from the work geoid to the home geoid'),
        geo_attrib('commuters_1251_to_3333_earnings', dtype=int, shape=Shapes.NxN,
                   comment='The number of total commuters that earn between $1251 to $3333 monthly from the work geoid to the home geoid'),
        geo_attrib('commuters_3333_over_earnings', dtype=int, shape=Shapes.NxN,
                   comment='The number of total commuters that earn more than $3333 monthly from the work geoid to the home geoid'),
        geo_attrib('commuters_goods_producing_industry', dtype=int, shape=Shapes.NxN,
                   comment='The number of total commuters that work in the goods producing industry from the work geoid to the home geoid'),
        geo_attrib('commuters_trade_transport_utility_industry', dtype=int, shape=Shapes.NxN,
                   comment='The number of total commuters that work in trade, transport, or utility industries from the work geoid to the home geoid'),
        geo_attrib('commuters_other_industry', dtype=int, shape=Shapes.NxN,
                   comment='The number of total commuters that work in any other industry than goods and producing, or trade, transport, or utilities from the work geoid to the home geoid'),
        geo_attrib('all_jobs', dtype=int, shape=Shapes.NxN,
                   comment='The number of total commuters that work in any type of job from the work geoid to the home geoid'),
        geo_attrib('primary_jobs', dtype=int, shape=Shapes.NxN,
                   comment='The number of total commuters that work only in primary jobs from the work geoid to the home geoid'),
        geo_attrib('all_private_jobs', dtype=int, shape=Shapes.NxN,
                   comment='The number of total commuters, all from private jobs, from the work geoid to the home geoid'),
        geo_attrib('private_primary_jobs', dtype=int, shape=Shapes.NxN,
                   comment='The number of total commuters that work in private primary jobs from the work geoid to the home geoid'),
        geo_attrib('all_federal_jobs', dtype=int, shape=Shapes.NxN,
                   comment='The number of total commuters, all from federal jobs, from the work geoid to the home geoid'),
        geo_attrib('federal_primary_jobs', dtype=int, shape=Shapes.NxN,
                   comment='The number of total commuters that work in federal primary jobs from the work geoid to the home geoid')
    ]

    attrib_vars = {
        'name': ['NAME'],
        'home_geoid': ["h_geocode"],
        'work_geoid': ['w_geocode'],
        'commuters': ["S000"],
        'commuters_29_under': ["SA01"],
        'commuters_30_to_54': ["SA02"],
        'commuters_55_over': ["SA03"],
        'commuters_goods_producing_industry': ["SI01"],
        'commuters_trade_transport_utility_industry': ["SI02"],
        'commuters_other_industry': ["SI03"],
        'all_jobs': ["JT00"],
        'primary_jobs': ["JT01"],
        'all_private_jobs': ["JT02"],
        'private_primary_jobs': ["JT03"],
        'all_federal_jobs': ["JT04"],
        'federal_primary_jobs': ["JT05"]
    }

    def make_adrio(self, attrib: AttributeDef, geography: Geography, time_period: TimePeriod) -> ADRIO:
        if attrib not in self.attributes:
            msg = f"{attrib.name} is not supported for the LODES data source."
            raise GeoValidationException(msg)
        if not isinstance(geography, LodesGeography):
            msg = f"LODES ADRIO requires LodesGeography, given {type(geography)}."
            raise GeoValidationException(msg)
        if not isinstance(time_period, Year):
            msg = f"LODES ADRIO requires Year (TimePeriod), given {type(time_period)}."
            raise GeoValidationException(msg)

        granularity = geography.granularity.value
        nodes = geography.filter
        year = time_period.year

        if attrib.name == 'primary_jobs':
            return self._make_commuter_adrio(granularity, nodes, 'S000', "JT01", year)
        elif attrib.name == 'all_private_jobs':
            return self._make_commuter_adrio(granularity, nodes, 'S000', "JT02", year)
        elif attrib.name == 'private_primary_jobs':
            return self._make_commuter_adrio(granularity, nodes, 'S000', "JT03", year)
        elif attrib.name == 'all_federal_jobs':
            return self._make_commuter_adrio(granularity, nodes, 'S000', "JT04", year)
        elif attrib.name == 'federal_primary_jobs':
            return self._make_commuter_adrio(granularity, nodes, 'S000', "JT05", year)
        elif attrib.name == 'commuters':
            return self._make_commuter_adrio(granularity, nodes, 'S000', "JT00", year)
        elif attrib.name == 'commuters_29_under':
            return self._make_commuter_adrio(granularity, nodes, 'SA01', "JT00", year)
        elif attrib.name == 'commuters_30_to_54':
            return self._make_commuter_adrio(granularity, nodes, 'SA02', "JT00", year)
        elif attrib.name == 'commuters_55_over':
            return self._make_commuter_adrio(granularity, nodes, 'SA03', "JT00", year)
        elif attrib.name == 'commuters_1250_under_earnings':
            return self._make_commuter_adrio(granularity, nodes, 'SE01', "JT00", year)
        elif attrib.name == 'commuters_1251_to_3333_earnings':
            return self._make_commuter_adrio(granularity, nodes, 'SE02', "JT00", year)
        elif attrib.name == 'commuters_3333_over_earnings':
            return self._make_commuter_adrio(granularity, nodes, 'SE03', "JT00", year)
        elif attrib.name == 'commuters_goods_producing_industry':
            return self._make_commuter_adrio(granularity, nodes, 'SI01', "JT00", year)
        elif attrib.name == 'commuters_trade_transport_utility_industry':
            return self._make_commuter_adrio(granularity, nodes, 'SI02', "JT00", year)
        elif attrib.name == 'commuters_other_industry':
            return self._make_commuter_adrio(granularity, nodes, 'SI03', "JT00", year)
        elif attrib.name == 'name':
            return self._make_name_adrio(granularity, nodes, "JT00", year)
        else:
            return super().make_adrio(attrib, geography, time_period)

    # fetch functions
    def _fetch_url(self, url: str) -> BytesIO:
        """Reads a file from a URL as a BytesIO."""
        with urlopen(url) as f:
            file_buffer = BytesIO()
            file_buffer.write(f.read())
            file_buffer.seek(0)
            return file_buffer

    def sort_geoids(self, granularity: str, list_geoids: list, geoids: tuple, data_frame: DataFrame):
        data_frame['w_geocode'] = data_frame['w_geocode'].astype(str)
        data_frame['h_geocode'] = data_frame['h_geocode'].astype(str)

        if not data_frame.empty:
            w_prefix = data_frame['w_geocode'].iloc[0][:2]
            h_prefix = data_frame['h_geocode'].iloc[0][:2]

            # if the file is a main file
            if w_prefix == h_prefix:
                if not geoids:
                    print("HERE")
                    # get the length depending on the granularity
                    if granularity == COUNTY.name:
                        length = COUNTY.length

                    elif granularity == TRACT.name:
                        length = TRACT.length

                    elif granularity == BLOCK.name:
                        length = BLOCK.length

                    elif granularity == BLOCK_GROUP.name:
                        length = BLOCK_GROUP.length

                    unique_geoids = data_frame['w_geocode'].apply(
                        lambda x: x[:length]).unique()

                    for geos in unique_geoids:
                        if geos not in list_geoids:
                            list_geoids.append(geos)

                    print(list_geoids)

                else:

                    # if the state for the geoid matches the main file and the geoid is not in the file, remove from the list
                    geoid_to_remove = [geo for geo in list_geoids if geo[:2] == w_prefix and not (
                        data_frame['w_geocode'].str.startswith(geo).any() or data_frame['h_geocode'].str.startswith(geo).any())]

                    for geos in geoid_to_remove:
                        list_geoids.remove(geos)

        return list_geoids

    # fetch files from LODES depending on the state, residence in/out of state, job type, and year
    def fetch_commuters(self, granularity: str, nodes: dict[str, list[str]], job_type: str, year: int):

        # file type is main (residence in state only) by default
        file_type = "main"

        state = copy(nodes.get('state')) or []

        # can change the lodes version, default is the most recent LODES8
        lodes_ver = "LODES8"

        data_frames = []

        # check for multiple states
        if (len(state) > 1):
            file_type = "aux"

        # check for valid years
        if year not in range(2002, 2022):
            passed_year = year
            # if before 2002 or after 2021, adjust to closest year
            if year in range(1999, 2002):
                year = 2002
            elif year in range(2022, 2025):
                year = 2021
            else:
                msg = "Invalid year. LODES data is only available for 2002-2021"
                raise Exception(msg)

            print(
                f"Commuting data cannot be retrieved for {passed_year}, fetching {year} data instead.")

        # if the year is between 2002-2009 and the job_type is JT04 or JT05
        if year in range(2002, 2010) and (job_type == "JT04" or job_type == "JT05"):

            # invalid, no federal jobs
            msg = "Invalid year for job type, no federal jobs can be found between 2002 to 2009"
            raise Exception(msg)

        invalid_conditions = [
            (year in range(2002, 2010) and (job_type == "JT04" or job_type == "JT05"),
             "Invalid year for job type, no federal jobs can be found between 2002 to 2009"),

            (('05' in state) and (year == 2002 or year in range(2019, 2022)),
             "Invalid year for state, no commuters can be found for Arkansas in 2002 or between 2019-2021"),

            (('04' in state) and (year == 2002 or year == 2003),
             "Invalid year for state, no commuters can be found for Arizona in 2002 or 2003"),

            (('11' in state) and (year in range(2002, 2010)),
             "Invalid year for state, no commuters can be found for DC in 2002 or between 2002-2009"),

            (('25' in state) and (year in range(2002, 2011)),
             "Invalid year for state, no commuters can be found for Massachusetts between 2002-2010"),

            (('28' in state) and (year in range(2002, 2004) or year in range(2019, 2022)),
             "Invalid year for state, no commuters can be found for Mississippi in 2002, 2003, or between 2019-2021"),

            (('33' in state) and year == 2002,
             "Invalid year for state, no commuters can be found for New Hampshire in 2002"),

            (('02' in state) and year in range(2017, 2022),
             "Invalid year for state, no commuters can be found for Alaska in between 2017-2021")
        ]
        for condition, message in invalid_conditions:
            if condition:
                raise Exception(message)

        # translate state FIPS code to state to use in URL
            # note, the if statement doesn't seem to be needed but gets rid of the None type error
        state_code_results = state_fips_to_code(2020)
        state_abbreviations = []
        for fips_code in state:
            state_code = state_code_results.get(fips_code).lower()
            state_abbreviations.append(state_code)

        # get the current geoid
        geoid = self.aggregate_geoid(granularity, nodes)
        list_geoids = list(geoid)

        for states in state_abbreviations:

            # construct the URL to fetch LODES data, reset to empty each time
            url_list = []
            cache_list = []
            files = []

            # always get main file (in state residency)
            url_main = f'https://lehd.ces.census.gov/data/lodes/{lodes_ver}/{states}/od/{states}_od_main_{job_type}_{year}.csv.gz'
            # print for testing purposes
            print(f"Fetching data from URL: {url_main}")
            url_list.append(url_main)

            # if there are more than one state in the input, get the aux files (out of state residence)
            if file_type == "aux":
                url_aux = f'https://lehd.ces.census.gov/data/lodes/{lodes_ver}/{states}/od/{states}_od_aux_{job_type}_{year}.csv.gz'
                print(f"Fetching data from URL: {url_main}")
                url_list.append(url_aux)
            # list the urls
            cache_list = [_LODES_CACHE_PATH / Path(u).name for u in url_list]

            # try to load the urls from the cache
            try:
                files = [load_file_from_cache(path) for path in cache_list]
                print("Load from cache\n")

            # on except CacheMiss
            except CacheMiss:

                print("Saving data to cache")

                # fetch info from the urls
                files = [_fetch_url(u) for u in url_list]

                # try to save the files
                try:
                    for f, path in zip(files, cache_list):
                        save_file_to_cache(path, f)

                except Exception as e:
                    msg = "We were unable to save LODES files to the cache.\n" \
                        f"Cause: {e}"
                    warn(msg, CacheWarning)

            unfiltered_df = []

            for file in files:
                data = pd.read_csv(file, compression="gzip", converters={
                                   'w_geocode': str, 'h_geocode': str})

                list_geoids = self.sort_geoids(granularity, list_geoids, geoid, data)

                unfiltered_df.append(data)

            # geoid = tuple(list_geoids)
            print(list_geoids, geoid)

            for df in unfiltered_df:

                filtered_rows_list = []

                # filter the rows on if they start with the prefix
                filtered_rows = df[df['h_geocode'].str.startswith(
                    tuple(list_geoids)) & df['w_geocode'].str.startswith(tuple(list_geoids))]

                filtered_rows_list.append(filtered_rows)

                # add the filtered dataframe to the list of dataframes
                data_frames.append(pd.concat(filtered_rows_list))

        return data_frames

    def _make_name_adrio(self, granularity: str, nodes: dict[str, list[str]], job_type: str, year: int) -> ADRIO:
        """Makes an ADRIO to retrieve home and work geocodes geoids."""
        def fetch() -> NDArray:
            # aggregate based on granularities
            geoid = self.aggregate_geoid(granularity, nodes)

            if geoid is None:
                geoid = ()
            list_geoids = list(geoid)

            data_frames = self.fetch_commuters(granularity, nodes, job_type, year)

            for df in data_frames:
                list_geoids = self.sort_geoids(granularity, list_geoids, geoid, df)

            geoid = tuple(list_geoids)

            # put together array of geoids
            n_geocode = len(geoid)
            output = np.empty((n_geocode), dtype=object)
            for w_id, w_geocode in enumerate(geoid):
                output[w_id] = (f"{w_geocode}")

            return np.array(output)

        return ADRIO('name', fetch)

    def _make_commuter_adrio(self, granularity: str, nodes: dict[str, list[str]], worker_type: str, job_type: str, year: int) -> ADRIO:
        """Makes an ADRIO to retrieve LODES commuting flow data."""
        def fetch() -> NDArray:
            data_frames = self.fetch_commuters(granularity, nodes, job_type, year)

            # initialize variables
            aggregated_data = None
            geoid = self.aggregate_geoid(granularity, nodes)
            list_geoids = list(geoid)

            # loop through the data frames and search for invalid geocodes
            for data_df in data_frames:
                list_geoids = self.sort_geoids(granularity, list_geoids, geoid, data_df)

            geoid = tuple(list_geoids)
            n_geocode = len(geoid)
            geocode_to_index = {geocode: i for i, geocode in enumerate(geoid)}
            geocode_len = len(geoid[0])

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
            for (w_geocode, h_geocode), value in aggregated_data.items():
                w_index = geocode_to_index.get(w_geocode)
                h_index = geocode_to_index.get(h_geocode)
                output[w_index, h_index] += np.int64(value)

            return output

        return ADRIO('commuters', fetch)

    def aggregate_geoid(self, granularity: str, nodes: dict[str, list[str]]):
        # get the value of the state
        state = copy(nodes.get('state'))

        # if no value of state
        if state == '*':
            # raise error
            msg = "Error: STATE value is not retrieved for state granularity"
            raise Exception(msg)

        home_geoids = []

        if state:
            home_geoids.extend(str(x) for x in state)

        # if the given aggregation is COUNTY
        # note: probably a more efficient way of doing this, but basically checks if the ADRIO needs that granularity to make geoids
        if granularity == Granularity.COUNTY.value or granularity != Granularity.STATE.value:
            county = copy(nodes.get('county'))

            # check if the current value is empty, if it is, return an empty tuple
            if county == ['*']:
                return ()

            if state and county:
                # if there is only one state
                home_geoids = [
                    state_code + county_code for state_code in state for county_code in county]

        # if the given aggregation is TRACT and that the tract value is not empty
        if granularity == Granularity.TRACT.value or granularity == Granularity.CBG.value or granularity == Granularity.BLOCK.value:
            tract = copy(nodes.get('tract'))

            print("TRACT")

            # check if the current value is empty
            if tract == ['*']:
                return ()

            if state and tract:

                home_geoids = [
                    geoid_code + tract_code for geoid_code in home_geoids for tract_code in tract]

        # if the given aggregation is CBG and that the CBG value is not empty
        if granularity == Granularity.CBG.value or granularity == Granularity.BLOCK.value:
            cbg = copy(nodes.get('block group'))

            # check if the current value is empty
            if cbg == ['*']:
                return ()

            if state and cbg:

                home_geoids = [
                    geoid_code + cbg_code for geoid_code in home_geoids for cbg_code in cbg]

        # if the given aggregation is block and the block value is not empty
        if granularity == Granularity.BLOCK.value:
            block = copy(nodes.get('block'))

            # check if the current value is empty
            if block == ['*']:
                return ()

            if state and block:
                home_geoids = [
                    geoid_code + block_code for geoid_code in home_geoids for block_code in block]

        return tuple(home_geoids)
