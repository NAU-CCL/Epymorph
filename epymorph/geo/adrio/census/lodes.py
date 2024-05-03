import gzip
import os
import shutil
import urllib.request
from copy import copy
from dataclasses import dataclass
from enum import Enum
from io import StringIO
from turtle import home

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plot_utils as pu
import us
from geopandas import GeoDataFrame
from numpy.typing import NDArray
from pandas import DataFrame, concat, read_excel
from pygris import tracts
from pygris.data import get_lodes
from sympy.core.symbol import Str

from epymorph.data_shape import Shapes
from epymorph.error import GeoValidationException
from epymorph.geo.adrio.adrio import ADRIO, ADRIOMaker
from epymorph.geo.spec import (AttribDef, CentroidDType, Geography, TimePeriod,
                               Year)


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

    CACHE_DIR = 'lodes_cache'

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

        # fetch files from LODES depending on the state, residence in/out of state, job type, and year
    def fetch_commuters(self, granularity: str, nodes: dict[str, list[str]], job_type: str, year: int):

        # notes for Tyler's code

        # check if the input state is a state
        # function: STATE.matches(#)

        # if not, throw error

        # otherwise, get state from FIPS code for the url
        # function: code_to_fips

        # or, if input as a state abbreviation, get the FIPS code for reading and comparing geocodes
        # function: state_fips_to_code?

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
                # question for clarification, if multiple states, are mapping inside the state to itself still wanted?

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

        # translate state FIPS code to state to use in URL
            # note, the if statement doesn't seem to be needed but gets rid of the None type error
        state_abbreviations = []
        if state:
            for states in state:
                state_name = pu._convert_FIPS_to_state_name({states: ''})
                state_name = pu._translate_state_abbrev(state_name, False)
                state_abbreviations.append(list(state_name.keys())[0].lower())

        # Tyler's code may be able to simplify this ^^

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

        # get the current geoid
        geoid = self.aggregate_geoid(granularity, nodes)

        # same here, may be able to rid of the aggregate_geoid function altogether

        for states in state_abbreviations:

            # construct the URL to fetch LODES data, reset to empty each time
            url_list = []
            cache_list = []

            # always get main file (in state residency)
            url = f'https://lehd.ces.census.gov/data/lodes/{lodes_ver}/{states}/od/{states}_od_main_{job_type}_{year}.csv.gz'
            cache_key = f"{year}_main_{states}_{job_type}"
            url_list.append(url)
            cache_list.append(cache_key)
            # print for testing purposes
            print(f"Fetching data from URL: {url}")

            # if there are more than one state in the input, get the aux files (out of state residence)
            if file_type is "aux":
                url_aux = f'https://lehd.ces.census.gov/data/lodes/{lodes_ver}/{states}/od/{states}_od_aux_{job_type}_{year}.csv.gz'
                cache_key = f"{year}_aux_{states}_{job_type}"
                url_list.append(url_aux)
                cache_list.append(cache_key)
                print(f"Fetching data from URL: {url_aux}")

            for urls, keys in zip(url_list, cache_list):
                cache_file_path = os.path.join(
                    self.CACHE_DIR, f"{keys}_{states}_{file_type}")
                if os.path.exists(cache_file_path):
                    print("Fetching data from cache...")
                    data = pd.read_csv(cache_file_path)
                else:
                    with urllib.request.urlopen(urls) as response:
                        with gzip.GzipFile(fileobj=response) as decompressed_file:
                            lodes_data = decompressed_file.read().decode('utf-8')

                    data = pd.read_csv(StringIO(lodes_data))

                    # cache the downloaded data
                    os.makedirs(self.CACHE_DIR, exist_ok=True)
                    data.to_csv(cache_file_path, index=False)

                    # check if there is a manner of doing this in the url ^^

                # make the compare strings for the home and work geocodes
                # based on granularity
                data['w_geocode'] = data['w_geocode'].astype(str).apply(
                    lambda x: x.zfill(15) if len(x) == 14 else x)
                data['h_geocode'] = data['h_geocode'].astype(str).apply(
                    lambda x: x.zfill(15) if len(x) == 14 else x)

                compare_h_geocode = data['h_geocode'].astype(str)
                compare_w_geocode = data['w_geocode'].astype(str)

                filtered_rows_list = []

                # filter the rows on if they start with the prefix
                filtered_rows = data[compare_h_geocode.str.startswith(
                    geoid) & compare_w_geocode.str.startswith(geoid)]

                # add the filtered rows to the list
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

            # if there is only one state
            if state:
                if county:
                    home_geoids = [
                        geoid_code + county_code for geoid_code in home_geoids for county_code in county]

                # otherwise, only add the same index to the current state

        # if the given aggregation is TRACT and that the tract value is not empty
        if granularity == Granularity.TRACT.value or granularity == Granularity.CBG.value or granularity == Granularity.BLOCK.value:
            tract = copy(nodes.get('tract'))

            # check if the current value is empty
            if tract == '*':
                # raise error
                msg = "Error: TRACT value is not retrieved for tract granularity"
                raise Exception(msg)

            if tract:
                home_geoids = [
                    geoid_code + tract_code for geoid_code in home_geoids for tract_code in tract]

        # if the given aggregation is CBG and that the CBG value is not empty
        if granularity == Granularity.CBG.value or granularity == Granularity.BLOCK.value:
            cbg = copy(nodes.get('block group'))

            # check if the current value is empty
            if cbg == '*':
                # raise error
                msg = "Error: CBG value is not retrieved for cbg granularity"
                raise Exception(msg)

            if cbg:
                home_geoids = [
                    geoid_code + cbg_code for geoid_code in home_geoids for cbg_code in cbg]

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

        return tuple(home_geoids)
