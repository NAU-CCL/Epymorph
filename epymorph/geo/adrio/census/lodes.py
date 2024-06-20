from pathlib import Path
from typing import Any
from urllib.request import urlopen
from warnings import warn

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from epymorph.cache import (CacheMiss, CacheWarning, load_file_from_cache,
                            save_file_to_cache)
from epymorph.data_shape import Shapes
from epymorph.error import GeoValidationException
from epymorph.geo.adrio.adrio import ADRIO, ADRIOMaker
from epymorph.geo.spec import TimePeriod, Year
from epymorph.geography.us_census import STATE, CensusScope, state_fips_to_code
from epymorph.geography.us_tiger import (_fetch_url, get_counties_info,
                                         get_states_info)
from epymorph.simulation import AttributeDef, geo_attrib

_LODES_CACHE_PATH = Path("geo/adrio/census/lodes")


class ADRIOMakerLODES(ADRIOMaker):
    """
    LODES8 ADRIO template to serve as parent class and provide utility functions for LODES8-based ADRIOS.
    """

    @staticmethod
    def accepts_source(source: Any):
        return False

    attributes = [
        geo_attrib('name', dtype=str, shape=Shapes.N,
                   comment='The proper name of the place.'),
        geo_attrib('geoid', dtype=str, shape=Shapes.N,
                   comment='The matrix of geoids from states or the input.'),
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
        'geoid': ["w_geocode"],
        'commuters': ["S000"],
        'commuters_29_under': ["SA01"],
        'commuters_30_to_54': ["SA02"],
        'commuters_55_over': ["SA03"],
        'commuters_1250_under_earnings': ["SE01"],
        'commuters_1251_to_3333_earnings': ["SE02"],
        'commuters_3333_over_earnings': ["SE03"],
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

    def make_adrio(self, attrib: AttributeDef, scope: CensusScope, time_period: TimePeriod) -> ADRIO:
        if attrib not in self.attributes:
            msg = f"{attrib.name} is not supported for the LODES data source."
            raise GeoValidationException(msg)
        if not isinstance(time_period, Year):
            msg = f"LODES ADRIO requires Year (TimePeriod), given {type(time_period)}."
            raise GeoValidationException(msg)

        year = time_period.year

        if attrib.name[:9] == "commuters":
            sorting_type = self.attrib_vars[attrib.name][0]
            return self._make_commuter_adrio(scope, sorting_type, "JT00", year)
        elif attrib.name[-4:] == "jobs":
            job_type = self.attrib_vars[attrib.name][0]
            return self._make_commuter_adrio(scope, "S000", job_type, year)
        elif attrib.name == 'name':
            return self._make_name_adrio(scope, "JT00", year)
        elif attrib.name == 'geoid':
            return self._make_geoid_adrio(scope, "JT00", year)
        else:
            return super().make_adrio(attrib, scope, time_period)

    # fetch files from LODES depending on the state, residence in/out of state, job type, and year
    def fetch_commuters(self, scope: CensusScope, job_type: str, year: int):

        # file type is main (residence in state only) by default
        file_type = "main"

        geoid = scope.get_node_ids()

        if scope.granularity != 'state':
            states = STATE.truncate_list(geoid)
        else:
            states = geoid

        # can change the lodes version, default is the most recent LODES8
        lodes_ver = "LODES8"

        data_frames = []

        # check for multiple states
        if (len(states) > 1):
            file_type = "aux"

        # check for valid years
        if year not in range(2002, 2022):
            passed_year = year
            # adjust to closest year
            if year in range(1999, 2002):
                year = 2002
            elif year in range(2022, 2025):
                year = 2021
            else:
                msg = "Invalid year. LODES data is only available for 2002-2021"
                raise Exception(msg)

            print(
                f"Commuting data cannot be retrieved for {passed_year}, fetching {year} data instead.")

        # no federal jobs in given years
        if year in range(2002, 2010) and (job_type == "JT04" or job_type == "JT05"):

            msg = "Invalid year for job type, no federal jobs can be found between 2002 to 2009"
            raise Exception(msg)

        # LODES year and state exceptions
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
                raise Exception(message)

        # translate state FIPS code to state to use in URL
        state_codes = state_fips_to_code(2020)
        state_abbreviations = [state_codes.get(
            fips, "").lower() for fips in states]

        for state in state_abbreviations:

            # construct the URL to fetch LODES data, reset to empty each time
            url_list = []
            cache_list = []
            files = []

            # always get main file (in state residency)
            url_main = f'https://lehd.ces.census.gov/data/lodes/{lodes_ver}/{state}/od/{state}_od_main_{job_type}_{year}.csv.gz'
            url_list.append(url_main)

            # if there are more than one state in the input, get the aux files (out of state residence)
            if file_type == "aux":
                url_aux = f'https://lehd.ces.census.gov/data/lodes/{lodes_ver}/{state}/od/{state}_od_aux_{job_type}_{year}.csv.gz'
                url_list.append(url_aux)
            cache_list = [_LODES_CACHE_PATH / Path(u).name for u in url_list]

            # try to load the urls from the cache
            try:
                files = [load_file_from_cache(path) for path in cache_list]

            # on except CacheMiss
            except CacheMiss:

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

            unfiltered_df = [pd.read_csv(file, compression="gzip", converters={
                'w_geocode': str, 'h_geocode': str}) for file in files]

            # go through dataframes, multiple if there are main and aux files
            for df in unfiltered_df:

                # filter the rows on if they start with the prefix
                filtered_rows = [df[df['h_geocode'].str.startswith(
                    tuple(geoid)) & df['w_geocode'].str.startswith(tuple(geoid))]]

                # add the filtered dataframe to the list of dataframes
                data_frames.append(pd.concat(filtered_rows))

        return data_frames

    def _make_name_adrio(self, scope: CensusScope, job_type: str, year: int) -> ADRIO:
        """Makes an ADRIO to retrieve the names of the states or counties given."""
        def fetch() -> np.ndarray:

            # can only fetch the name information for state or county granularity, returns an empty array otherwise
            if scope.granularity == 'state':
                name_info = get_states_info(2020)
            elif scope.granularity == 'county':
                name_info = get_counties_info(2020)
            else:
                return np.array([], dtype=str)

            geoid = scope.get_node_ids()

            geoid_to_name = dict(zip(name_info['GEOID'], name_info['NAME']))

            county_names = [geoid_to_name.get(geo, "Unknown") for geo in geoid]

            output = np.array(county_names, dtype=str)

            return output

        return ADRIO('name', fetch)

    def _make_geoid_adrio(self, scope: CensusScope, job_type: str, year: int) -> ADRIO:
        """Makes an ADRIO to retrieve home and work geocodes geoids."""
        def fetch() -> NDArray:

            geoid = scope.get_node_ids()
            output = np.array(geoid, dtype=str)

            return output

        return ADRIO('geoid', fetch)

    def _make_commuter_adrio(self, scope: CensusScope, worker_type: str, job_type: str, year: int) -> ADRIO:
        """Makes an ADRIO to retrieve LODES commuting flow data."""
        def fetch() -> NDArray:
            data_frames = self.fetch_commuters(scope, job_type, year)

            # initialize variables
            aggregated_data = None
            geoid = scope.get_node_ids()
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
            for (w_geocode, h_geocode), value in aggregated_data.items():  # type: ignore
                w_index = geocode_to_index.get(w_geocode)
                h_index = geocode_to_index.get(h_geocode)
                output[w_index, h_index] += np.int64(value)

            return output

        return ADRIO('commuters', fetch)
