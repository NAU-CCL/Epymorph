from copy import copy
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path
from turtle import home
from urllib.request import urlopen
from warnings import warn

import geopandas
import matplotlib.pyplot
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
from epymorph.geo.spec import AttribDef, Geography, TimePeriod, Year
from epymorph.geography.us_census import state_fips_to_code
from epymorph.geography.us_tiger import _fetch_url

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
        AttribDef('name', np.str_, Shapes.N),
        AttribDef('home_geoid', np.str_, Shapes.N),
        AttribDef('work_geoid', np.str_, Shapes.N),
        AttribDef('commuters', np.int64, Shapes.NxN),
        AttribDef('commuters_29_under', np.int64, Shapes.NxN),
        AttribDef('commuters_30_to_54', np.int64, Shapes.NxN),
        AttribDef('commuters_55_over', np.int64, Shapes.NxN),
        AttribDef('commuters_1250_under_earnings', np.int64, Shapes.NxN),
        AttribDef('commuters_1251_to_3333_earnings', np.int64, Shapes.NxN),
        AttribDef('commuters_3333_over_earnings', np.int64, Shapes.NxN),
        AttribDef('commuters_goods_producing_industry', np.int64, Shapes.NxN),
        AttribDef('commuters_trade_transport_utility_industry', np.int64, Shapes.NxN),
        AttribDef('commuters_other_industry', np.int64, Shapes.NxN),
        AttribDef('all_jobs', np.int64, Shapes.NxN),
        AttribDef('primary_jobs', np.int64, Shapes.NxN),
        AttribDef('all_private_jobs', np.int64, Shapes.NxN),
        AttribDef('private_primary_jobs', np.int64, Shapes.NxN),
        AttribDef('all_federal_jobs', np.int64, Shapes.NxN),
        AttribDef('federal_primary_jobs', np.int64, Shapes.NxN)
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

    def make_adrio(self, attrib: AttribDef, geography: Geography, time_period: TimePeriod) -> ADRIO:
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
            return self._make_name_adrio(granularity, nodes, year)
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

    # fetch files from LODES depending on the state, residence in/out of state, job type, and year
    def fetch_commuters(self, granularity: str, nodes: dict[str, list[str]], job_type: str, year: int):

        # file type is main (residence in state only) by default
        file_type = "main"

        # fetch the state for the URL
        state = copy(nodes.get('state'))

        # can change the lodes version, default is the most recent LODES8
        lodes_ver = "LODES8"

        data_frames = []

        # check for multiple states
        if state:
            if (len(state) > 1):

                # if multiple states, automatically aux file (residence out of state)
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

        # check for valid files

        # if the year is between 2002-2009 and the job_type is JT04 or JT05
        if year in range(2002, 2010) and (job_type == "JT04" or job_type == "JT05"):

            # invalid, no federal jobs
            msg = "Invalid year for job type, no federal jobs can be found between 2002 to 2009"
            raise Exception(msg)

        if state:
            # if the state is 05 and the year is 2002 or between 2019-2021
            if ('05' in state) and (year == 2002 or year in range(2019, 2022)):

                msg = "Invalid year for state, no commuters can be found for Arkansas in 2002 or between 2019-2021"
                raise Exception(msg)

            # if the state is 04 and the year is 2002 or 2003
            if ('04' in state) and (year == 2002 or year == 2003):

                msg = "Invalid year for state, no commuters can be found for Arizona in 2002 or 2003"
                raise Exception(msg)

            # if the state is 11 and the year is between 2002-2009
            if ('11' in state) and (year in range(2002, 2010)):

                msg = "Invalid year for state, no commuters can be found for DC in 2002 or between 2002-2009"
                raise Exception(msg)

            # if the state is 25 and the year is 2002-2010
            if ('25' in state) and (year in range(2002, 2011)):

                msg = "Invalid year for state, no commuters can be found for Massachusetts between 2002-2010"
                raise Exception(msg)

            # if the state is 28 and the year is 2002, 2003, or between 2019-2021
            if ('28' in state) and (year in range(2002, 2004) or year in range(2019, 2022)):

                msg = "Invalid year for state, no commuters can be found for Mississippi in 2002, 2003, or between 2019-2021"
                raise Exception(msg)

            # if the state is 33 and the year is 2002
            if ('33' in state) and year == 2002:

                msg = "Invalid year for state, no commuters can be found for New Hampshire in 2002"
                raise Exception(msg)

            # if the state is 02 and the year is 2017, 2018, or between 2019-2021
            if ('02' in state) and year in range(2017, 2022):

                msg = "Invalid year for state, no commuters can be found for Alaska in between 2017-2021"
                raise Exception(msg)

        # translate state FIPS code to state to use in URL
            # note, the if statement doesn't seem to be needed but gets rid of the None type error
        state_code_results = state_fips_to_code(2020)
        state_abbreviations = []
        if state:
            for fips_code in state:
                state_code = state_code_results.get(fips_code).lower()
                state_abbreviations.append(state_code)

        # get the current geoid
        geoid = self.aggregate_geoid(granularity, nodes)

        # loop through to check for

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

            for file in files:
                data = pd.read_csv(file, compression="gzip", converters={
                                   'w_geocode': str, 'h_geocode': str})

                filtered_rows_list = []

                # filter the rows on if they start with the prefix
                filtered_rows = data[data['h_geocode'].str.startswith(
                    geoid) & data['w_geocode'].str.startswith(geoid)]

                filtered_rows_list.append(filtered_rows)

                # add the filtered dataframe to the list of dataframes
                data_frames.append(pd.concat(filtered_rows_list))

        return data_frames

    def _make_name_adrio(self, granularity: str, nodes: dict[str, list[str]], year: int) -> ADRIO:
        """Makes an ADRIO to retrieve home and work geocodes geoids."""
        def fetch() -> NDArray:
            # aggregate based on granularities
            geoid = self.aggregate_geoid(granularity, nodes)

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
            geocode_len = len(geoid[0])
            n_geocode = len(geoid)
            geocode_to_index = {geocode: i for i, geocode in enumerate(geoid)}

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

            # check if the current value is empty
            if county == '*':
                # raise error
                msg = "Error: COUNTY value is not retrieved for county granularity"
                raise Exception(msg)

            if state and county:
                # if there is only one state
                if len(state) == 1:
                    home_geoids = [
                        state_code + county_code for state_code in state for county_code in county]
                else:
                    # aggregate based on index if there are multiple states
                    home_geoids = [state[i] + county[i]
                                   for i in range(min(len(state), len(county)))]

        # if the given aggregation is TRACT and that the tract value is not empty
        if granularity == Granularity.TRACT.value or granularity == Granularity.CBG.value or granularity == Granularity.BLOCK.value:
            tract = copy(nodes.get('tract'))

            # check if the current value is empty
            if tract == '*':
                # raise error
                msg = "Error: TRACT value is not retrieved for tract granularity"
                raise Exception(msg)

            if state and tract:
                # if there is only one state
                if len(state) == 1:
                    home_geoids = [
                        geoid_code + tract_code for geoid_code in home_geoids for tract_code in tract]
                else:
                    # aggregate based on index if there are multiple states
                    home_geoids = [home_geoids[i] + tract[i]
                                   for i in range(min(len(home_geoids), len(tract)))]

        # if the given aggregation is CBG and that the CBG value is not empty
        if granularity == Granularity.CBG.value or granularity == Granularity.BLOCK.value:
            cbg = copy(nodes.get('block group'))

            # check if the current value is empty
            if cbg == '*':
                # raise error
                msg = "Error: CBG value is not retrieved for cbg granularity"
                raise Exception(msg)

            if state and cbg:
                # if there is only one state
                if len(state) == 1:
                    home_geoids = [
                        geoid_code + cbg_code for geoid_code in home_geoids for cbg_code in cbg]
                else:
                    # aggregate based on index if there are multiple states
                    home_geoids = [home_geoids[i] + cbg[i]
                                   for i in range(min(len(home_geoids), len(cbg)))]

        # if the given aggregation is block and the block value is not empty
        if granularity == Granularity.BLOCK.value:
            block = copy(nodes.get('block'))

            # check if the current value is empty
            if block == '*':
                # raise error
                msg = "Error: BLOCK value is not retrieved for block granularity"
                raise Exception(msg)

            if block:
                home_geoids = [
                    geoid_code + block_code for geoid_code in home_geoids for block_code in block]

            if state and block:
                # if there is only one state
                if len(state) == 1:
                    home_geoids = [
                        geoid_code + block_code for geoid_code in home_geoids for block_code in block]
                else:
                    # aggregate based on index if there are multiple states
                    home_geoids = [home_geoids[i] + block[i]
                                   for i in range(min(len(home_geoids), len(block)))]

        return tuple(home_geoids)
