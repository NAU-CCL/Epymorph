import os
from copy import copy
from dataclasses import dataclass
from enum import Enum

import numpy as np
from census import Census
from geopandas import GeoDataFrame
from numpy.typing import NDArray
from pandas import DataFrame, concat, read_excel
from pygris import block_groups, counties, states, tracts

from epymorph.data_shape import Shapes
from epymorph.error import GeoValidationException
from epymorph.geo.adrio.adrio import ADRIO, ADRIOMaker
from epymorph.geo.spec import (AttribDef, CentroidDType, Geography, TimePeriod,
                               Year)


class Granularity(Enum):
    """Enumeration of Census granularity levels."""
    STATE = 0
    COUNTY = 1
    TRACT = 2
    CBG = 3


@dataclass
class CensusGeography(Geography):
    """Dataclass used to describe geographies for Census ADRIOs."""

    granularity: Granularity
    """The granularity level to fetch data from."""

    filter: dict[str, list[str]]
    """A list of locations to fetch data from as FIPS codes for relevant granularities."""


class ADRIOMakerCensus(ADRIOMaker):
    """
    Census ADRIO template to serve as parent class and provide utility functions for Census-based ADRIOS.
    """

    population_query = ['B01001_003E',  # population 0-19
                        'B01001_004E',
                        'B01001_005E',
                        'B01001_006E',
                        'B01001_007E',
                        'B01001_027E',  # women
                        'B01001_028E',
                        'B01001_029E',
                        'B01001_030E',
                        'B01001_031E',
                        'B01001_008E',  # population 20-34
                        'B01001_009E',
                        'B01001_010E',
                        'B01001_011E',
                        'B01001_012E',
                        'B01001_032E',  # women
                        'B01001_033E',
                        'B01001_034E',
                        'B01001_035E',
                        'B01001_036E',
                        'B01001_013E',  # population 35-54
                        'B01001_014E',
                        'B01001_015E',
                        'B01001_016E',
                        'B01001_037E',  # women
                        'B01001_038E',
                        'B01001_039E',
                        'B01001_040E',
                        'B01001_017E',  # population 55-64
                        'B01001_018E',
                        'B01001_019E',
                        'B01001_041E',  # women
                        'B01001_042E',
                        'B01001_043E',
                        'B01001_020E',  # population 65-74
                        'B01001_021E',
                        'B01001_022E',
                        'B01001_044E',  # women
                        'B01001_045E',
                        'B01001_046E',
                        'B01001_023E',  # population 75+
                        'B01001_024E',
                        'B01001_025E',
                        'B01001_047E',  # women
                        'B01001_048E',
                        'B01001_049E']

    attributes = [
        AttribDef('name', np.str_, Shapes.N),
        AttribDef('population', np.int64, Shapes.N),
        AttribDef('population_by_age', np.int64, Shapes.NxA(3)),
        AttribDef('population_by_age_x6', np.int64, Shapes.NxA(6)),
        AttribDef('centroid', CentroidDType, Shapes.N),
        AttribDef('geoid', np.str_, Shapes.N),
        AttribDef('average_household_size', np.int64, Shapes.N),
        AttribDef('dissimilarity_index', np.float64, Shapes.N),
        AttribDef('commuters', np.int64, Shapes.NxN),
        AttribDef('gini_index', np.float64, Shapes.N),
        AttribDef('median_age', np.int64, Shapes.N),
        AttribDef('median_income', np.int64, Shapes.N),
        AttribDef('tract_median_income', np.int64, Shapes.N),
        AttribDef('pop_density_km2', np.float64, Shapes.N),
    ]

    attrib_vars = {
        'name': ['NAME'],
        'geoid': ['NAME'],
        'population': ['B01001_001E'],
        'population_by_age': population_query,
        'population_by_age_x6': population_query,
        'median_income': ['B19013_001E'],
        'median_age': ['B01002_001E'],
        'tract_median_income': ['B19013_001E'],
        'dissimilarity_index': ['B03002_003E',  # white population
                                'B03002_013E',
                                'B03002_004E',  # minority population
                                'B03002_014E'],
        'average_household_size': ['B25010_001E'],
        'gini_index': ['B19083_001E'],
        'pop_density_km2': ['B01003_001E'],
    }

    census: Census
    """Census API interface object."""

    def __init__(self) -> None:
        """Initializer to create Census object."""
        api_key = os.environ.get('CENSUS_API_KEY')
        if api_key is None:
            msg = "Census API key not found. Please ensure you have an API key and have assigned it to an environment variable named 'CENSUS_API_KEY'"
            raise Exception(msg)
        self.census = Census(api_key)

    def make_adrio(self, attrib: AttribDef, geography: Geography, time_period: TimePeriod) -> ADRIO:
        if attrib not in self.attributes:
            msg = f"{attrib.name} is not supported for the Census data source."
            raise GeoValidationException(msg)
        if not isinstance(geography, CensusGeography):
            msg = f"Census ADRIO requires CensusGeography, given {type(geography)}."
            raise GeoValidationException(msg)
        if not isinstance(time_period, Year):
            msg = f"Census ADRIO requires Year (TimePeriod), given {type(time_period)}."
            raise GeoValidationException(msg)

        granularity = geography.granularity.value
        nodes = geography.filter
        year = time_period.year

        if attrib.name == 'geoid':
            return self._make_geoid_adrio(granularity, nodes, year)
        elif attrib.name == 'population_by_age':
            return self._make_population_adrio(granularity, nodes, year, 3)
        elif attrib.name == 'dissimilarity_index':
            return self._make_dissimilarity_index_adrio(granularity, nodes, year)
        elif attrib.name == 'gini_index':
            return self._make_gini_index_adrio(granularity, nodes, year)
        elif attrib.name == 'pop_density_km2':
            return self._make_pop_density_adrio(granularity, nodes, year)
        elif attrib.name == 'centroid':
            return self._make_centroid_adrio(granularity, nodes, year)
        elif attrib.name == 'tract_median_income':
            return self._make_tract_med_income_adrio(nodes, year)
        elif attrib.name == 'commuters':
            return self._make_commuter_adrio(granularity, nodes, year)
        else:
            return self._make_simple_adrios(attrib, granularity, nodes, year)

    def fetch_acs5(self, variables: list[str], granularity: int, nodes: dict[str, list[str]], year: int) -> DataFrame:
        """Utility function to fetch Census data by building queries from ADRIO data."""
        # verify node types and convert to strings usable by census api
        states = nodes.get('state')
        counties = nodes.get('county')
        tracts = nodes.get('tract')
        cbg = nodes.get('block group')
        if type(states) is list:
            states = ','.join(states)
        if type(counties) is list:
            counties = ','.join(counties)
        if type(tracts) is list:
            tracts = ','.join(tracts)
        if type(cbg) is list:
            cbg = ','.join(cbg)

        # fetch and sort data according to granularity
        if granularity == Granularity.STATE.value:
            data = self.census.acs5.get(
                variables, {'for': f'state: {states}'}, year=year)
            sort_param = ['state']
        elif granularity == Granularity.COUNTY.value:
            data = self.census.acs5.get(
                variables, {'for': f'county: {counties}', 'in': f'state: {states}'}, year=year)
            sort_param = ['state', 'county']
        elif granularity == Granularity.TRACT.value:
            data = self.census.acs5.get(variables, {
                                        'for': f'tract: {tracts}', 'in': f'state: {states} county: {counties}'}, year=year)
            sort_param = ['state', 'county', 'tract']
        else:
            data = self.census.acs5.get(variables, {
                                        'for': f'block group: {cbg}', 'in': f'state: {states} county: {counties} tract: {tracts}'}, year=year)
            sort_param = ['state', 'county', 'tract', 'block group']

        data_df = DataFrame.from_records(data)

        data_df = data_df.sort_values(by=sort_param)
        data_df.reset_index(inplace=True)

        # return data to adrio for processing
        return data_df

    def fetch_sf(self, granularity: int, nodes: dict[str, list[str]], year: int) -> GeoDataFrame:
        """Utility function to fetch shape files from Census for specified regions."""
        state_fips = nodes.get('state')
        county_fips = nodes.get('county')
        tract_fips = nodes.get('tract')
        cbg_fips = nodes.get('block group')

        # call appropriate pygris function based on granularity and sort result
        if granularity == Granularity.STATE.value:
            data_df = states(year=year)
            data_df = data_df.rename(columns={'STATEFP': 'state'})

            if state_fips is not None and state_fips[0] != '*':
                data_df = data_df.loc[data_df['state'].isin(state_fips)]

            sort_param = ['state']

        elif granularity == Granularity.COUNTY.value:
            if state_fips is None or '*' in state_fips:
                data_df = counties(year=year)
            else:
                data_df = counties(state=state_fips, year=year)
            data_df = data_df.rename(columns={'STATEFP': 'state', 'COUNTYFP': 'county'})

            if county_fips is not None and county_fips[0] != '*':
                data_df = data_df.loc[data_df['county'].isin(county_fips)]

            sort_param = ['state', 'county']

        elif granularity == Granularity.TRACT.value:
            if state_fips is not None and state_fips[0] != '*' and county_fips is not None and county_fips[0] != '*':
                data_df = GeoDataFrame()
                # tract and block group level files cannot be fetched using lists
                # several queries must be made and merged instead
                for i in range(len(state_fips)):
                    for j in range(len(county_fips)):
                        current_data = tracts(
                            state=state_fips[i], county=county_fips[j], year=year)

                        if len(data_df.index) > 0:
                            data_df = data_df.merge(
                                current_data, on=['STATEFP', 'COUNTYFP', 'TRACTCE'])
                        else:
                            data_df = current_data
            else:
                msg = "Data could not be retrieved due to missing state or county fips codes. \
                Wildcard specifier(*) cannot be used for tract level data."
                raise Exception(msg)

            data_df = data_df.rename(
                columns={'STATEFP': 'state', 'COUNTYFP': 'county', 'TRACTCE': 'tract'})

            if tract_fips is not None and tract_fips[0] != '*':
                data_df = data_df.loc[data_df['tract'].isin(tract_fips)]

            sort_param = ['state', 'county', 'tract']

        else:
            state_fips = nodes.get('state')
            county_fips = nodes.get('county')
            if state_fips is not None and state_fips[0] != '*' and county_fips is not None and county_fips[0] != '*':
                data_df = GeoDataFrame()
                for i in range(len(state_fips)):
                    for j in range(len(county_fips)):
                        current_data = block_groups(
                            state=state_fips[i], county=county_fips[j], year=year)
                        if len(data_df.index) > 0:
                            data_df = data_df.merge(
                                current_data, on=['STATEFP', 'COUNTYFP', 'TRACTCE', 'BLKGRPCE'])
                        else:
                            data_df = current_data
            else:
                msg = "Data could not be retrieved due to missing state or county fips codes. \
                    Wildcard specifier(*) cannot be used for block group level data."
                raise Exception(msg)

            data_df = data_df.rename(
                columns={'STATEFP': 'state', 'COUNTYFP': 'county', 'TRACTCE': 'tract', 'BLKGRPCE': 'block group'})
            if cbg_fips is not None and cbg_fips[0] != '*':
                data_df = data_df.iloc[data_df['block group'].isin(cbg_fips)]

            sort_param = ['state', 'county', 'block group']

        data_df = GeoDataFrame(data_df.sort_values(by=sort_param))
        data_df.reset_index(drop=True, inplace=True)

        return data_df

    def fetch_commuters(self, granularity: int, nodes: dict[str, list[str]], year: int) -> DataFrame:
        """
        Utility function to fetch commuting data from .xslx format filtered down to requested regions.
        """
        # check for invalid granularity
        if granularity == Granularity.CBG.value or granularity == Granularity.TRACT.value:
            msg = "Error: Commuting data cannot be retrieved for tract or block group granularities"
            raise Exception(msg)

        # check for valid year
        if year not in [2010, 2015, 2020]:
            # if invalid year is close to a valid year, fetch valid data and notify user
            passed_year = year
            if year in range(2008, 2012):
                year = 2010
            elif year in range(2013, 2017):
                year = 2015
            elif year in range(2018, 2022):
                year = 2020
            else:
                msg = "Invalid year. Communting data is only available for 2008-2022"
                raise Exception(msg)

            print(
                f"Commuting data cannot be retrieved for {passed_year}, fetching {year} data instead.")

        states = copy(nodes.get('state'))
        counties = copy(nodes.get('county'))

        if year != 2010:
            url = f'https://www2.census.gov/programs-surveys/demo/tables/metro-micro/{year}/commuting-flows-{year}/table1.xlsx'

            # organize dataframe column names
            group_fields = ['state_code',
                            'county_code',
                            'state',
                            'county']

            all_fields = ['res_' + field for field in group_fields] + \
                         ['wrk_' + field for field in group_fields] + \
                         ['workers', 'moe']

            header_num = 6

        else:
            url = 'https://www2.census.gov/programs-surveys/demo/tables/metro-micro/2010/commuting-employment-2010/table1.xlsx'

            all_fields = ['res_state_code', 'res_county_code', 'wrk_state_code', 'wrk_county_code',
                          'workers', 'moe', 'res_state', 'res_county', 'wrk_state', 'wrk_county']

            header_num = 3

        # download communter data spreadsheet as a pandas dataframe
        data = read_excel(url, header=header_num, names=all_fields, dtype={
                          'res_state_code': str, 'wrk_state_code': str, 'res_county_code': str, 'wrk_county_code': str})

        if states is not None:
            # states specified
            if states[0] != '*':
                data = data.loc[data['res_state_code'].isin(states)]

                for i in range(len(states)):
                    states[i] = states[i].zfill(3)
                data = data.loc[data['wrk_state_code'].isin(states)]

            # wildcard case
            else:
                data = data.loc[data['res_state_code'] < '57']
                data = data.loc[data['res_state_code'] != '11']
                data = data.loc[data['wrk_state_code'] < '057']
                data = data.loc[data['wrk_state_code'] != '011']

            # filter out non-county locations
            data = data.loc[data['res_county_code'] < '508']
            data = data.loc[data['wrk_county_code'] < '508']

        if granularity == Granularity.COUNTY.value:
            if counties is not None and counties[0] != '*':
                data = data.loc[data['res_county_code'].isin(counties)]
                data = data.loc[data['wrk_county_code'].isin(counties)]

        return data

    def _make_geoid_adrio(self, granularity: int, nodes: dict[str, list[str]], year: int) -> ADRIO:
        """Makes an ADRIO to retrieve GEOID."""
        def fetch() -> NDArray:
            data_df = self.fetch_acs5(
                self.attrib_vars['geoid'], granularity, nodes, year)
            # strange interaction here - name field is fetched only because a field is required
            data_df = data_df.drop(columns='NAME')

            # concatenate individual fips codes to yield geoid
            output = list()
            for i in range(len(data_df.index)):
                # state geoid is the same as fips code - no action required
                if granularity == Granularity.STATE.value:
                    output.append(str(data_df.loc[i, 'state']))
                elif granularity == Granularity.COUNTY.value:
                    output.append(str(data_df.loc[i, 'state']) +
                                  str(data_df.loc[i, 'county']))
                elif granularity == Granularity.TRACT.value:
                    output.append(str(
                        data_df.loc[i, 'state']) + str(data_df.loc[i, 'county']) + str(data_df.loc[i, 'tract']))
                else:
                    output.append(str(data_df.loc[i, 'state']) + str(data_df.loc[i, 'county']) + str(
                        data_df.loc[i, 'tract']) + str(data_df.loc[i, 'block group']))

            return np.array(output, dtype=np.str_)
        return ADRIO('geoid', fetch)

    def _make_population_adrio(self, granularity: int, nodes: dict[str, list[str]], year: int, num_groups: int) -> ADRIO:
        """Makes an ADRIO to retrieve population data split into 3 or 6 age groups."""
        def fetch() -> NDArray:
            data_df = self.fetch_acs5(self.population_query, granularity, nodes, year)
            # calculate population of each age bracket and enter into a numpy array to return
            output = np.zeros((len(data_df.index), num_groups), dtype=np.int64)
            pop = [0, 0, 0, 0, 0, 0]
            for i in range(len(data_df.index)):
                for j in range(len(data_df.iloc[i].index)):
                    if j >= 0 and j < 10:
                        pop[0] += data_df.iloc[i].iloc[j]
                    elif j >= 10 and j < 20:
                        pop[1] += data_df.iloc[i].iloc[j]
                    elif j >= 20 and j < 28:
                        pop[2] += data_df.iloc[i].iloc[j]
                    elif j >= 28 and j < 34:
                        pop[3] += data_df.iloc[i].iloc[j]
                    elif j >= 34 and j < 40:
                        pop[4] += data_df.iloc[i].iloc[j]
                    elif j < 47:
                        pop[5] += data_df.iloc[i].iloc[j]

                if num_groups == 3:
                    output[i] = [pop[0], pop[1] + pop[2] + pop[3], pop[4] + pop[5]]
                else:
                    output[i] = pop

            return output

        if num_groups == 3:
            return ADRIO('population_by_age', fetch)
        else:
            return ADRIO('population_by_age_x6', fetch)

    def _make_dissimilarity_index_adrio(self, granularity: int, nodes: dict[str, list[str]], year: int) -> ADRIO:
        """Makes an ADRIO to retrieve dissimilarity index."""
        def fetch() -> NDArray:
            data_df = self.fetch_acs5(
                self.attrib_vars['dissimilarity_index'], granularity, nodes, year)
            data_df2 = self.fetch_acs5(
                self.attrib_vars['dissimilarity_index'], granularity + 1, nodes, year)

            output = np.zeros(len(data_df2.index), dtype=np.float64)

            # loop for counties
            j = 0
            for i in range(len(data_df2.index)):
                # assign county fip to variable
                county_fip = data_df2.iloc[i][str(
                    Granularity(granularity).name).lower()]
                # loop for all tracts in county (while fip == variable)
                sum = 0.0
                while data_df.iloc[j][str(Granularity(granularity).name).lower()] == county_fip and j < len(data_df.index) - 1:
                    # preliminary calculations
                    tract_minority = data_df.iloc[j]['B03002_004E'] + \
                        data_df.iloc[j]['B03002_014E']
                    county_minority = data_df2.iloc[i]['B03002_004E'] + \
                        data_df2.iloc[i]['B03002_014E']
                    tract_majority = data_df.iloc[j]['B03002_003E'] + \
                        data_df.iloc[j]['B03002_013E']
                    county_majority = data_df2.iloc[i]['B03002_003E'] + \
                        data_df2.iloc[i]['B03002_013E']

                    # run calculation sum += ( |minority(tract) / minority(county) - majority(tract) / majority(county)| )
                    if county_minority != 0 and county_majority != 0:
                        sum = sum + abs(tract_minority / county_minority -
                                        tract_majority / county_majority)
                    j += 1

                sum *= .5
                if sum == 0.:
                    sum = 0.5

                # assign current output element to sum
                output[i] = sum

            return output
        return ADRIO('dissimilarity_index', fetch)

    def _make_gini_index_adrio(self, granularity: int, nodes: dict[str, list[str]], year: int) -> ADRIO:
        """Makes an ADRIO to retrieve gini index."""
        def fetch() -> NDArray:
            data_df = self.fetch_acs5(
                self.attrib_vars['gini_index'], granularity, nodes, year)
            data_df2 = None
            data_df['B19083_001E'] = data_df['B19083_001E'].astype(
                np.float64).fillna(0.5).replace(-666666666, 0.5)

            # set cbg data to that of the parent tract if geo granularity = cbg
            if granularity == Granularity.CBG.value:
                print(
                    'Gini Index cannot be retrieved for block group level, fetching tract level data instead.')
                data_df2 = self.fetch_acs5(
                    self.attrib_vars['gini_index'], granularity - 1, nodes, year)
                j = 0
                for i in range(len(data_df.index)):
                    tract_fip = data_df.loc[i, 'tract']
                    while data_df2.loc[j, 'tract'] == tract_fip and j < len(data_df2.index) - 1:
                        data_df2.loc[j, 'B01001_001E'] = data_df.loc[i, 'B19083_001E']
                        j += 1
                data_df = data_df2
            return data_df[self.attrib_vars['gini_index']].to_numpy(dtype=np.float64).squeeze()
        return ADRIO('gini_index', fetch)

    def _make_pop_density_adrio(self, granularity: int, nodes: dict[str, list[str]], year: int) -> ADRIO:
        """Makes an ADRIO to retrieve population density per km2."""
        def fetch() -> NDArray:
            data_df = self.fetch_acs5(
                self.attrib_vars['pop_density_km2'], granularity, nodes, year)
            geo_df = self.fetch_sf(granularity, nodes, year)
            # merge census data with shapefile data
            if granularity == Granularity.STATE.value:
                geo_df = geo_df.merge(data_df, on=['state'])
            elif granularity == Granularity.COUNTY.value:
                geo_df = geo_df.merge(data_df, on=['state', 'county'])
            elif granularity == Granularity.TRACT.value:
                geo_df = geo_df.merge(data_df, on=['state', 'county', 'tract'])
            else:
                geo_df = geo_df.merge(
                    data_df, on=['state', 'county', 'tract', 'block group'])

            # calculate population density, storing it in a numpy array to return
            output = np.zeros(len(geo_df.index), dtype=np.float64)
            for i in range(len(geo_df.index)):
                output[i] = round(int(geo_df.iloc[i]['B01003_001E']) /
                                  (geo_df.iloc[i]['ALAND'] / 1e6))
            return output
        return ADRIO('pop_density_km2', fetch)

    def _make_centroid_adrio(self, granularity: int, nodes: dict[str, list[str]], year: int):
        """Makes an ADRIO to retrieve geographic centroid coordinates."""
        def fetch() -> NDArray:
            data_df = self.fetch_sf(granularity, nodes, year)
            # map node's name to its centroid in a numpy array and return
            output = np.zeros(len(data_df.index), dtype=CentroidDType)
            for i in range(len(data_df.index)):
                output[i] = data_df.iloc[i]['geometry'].centroid.coords[0]

            return output
        return ADRIO('centroid', fetch)

    def _make_tract_med_income_adrio(self, nodes: dict[str, list[str]], year: int) -> ADRIO:
        """Makes an ADRIO to retrieve median income at the Census tract level."""
        def fetch() -> NDArray:
            data_df = self.fetch_acs5(
                self.attrib_vars['median_income'], Granularity.TRACT.value, nodes, year)
            data_df2 = self.fetch_acs5(
                self.attrib_vars['median_income'], Granularity.CBG.value, nodes, year)
            data_df = data_df.fillna(0).replace(-666666666, 0)
            # set cbg data to that of the parent tract
            j = 0
            for i in range(len(data_df.index)):
                tract_fip = data_df.loc[i, 'tract']
                while data_df2.loc[j, 'tract'] == tract_fip and j < len(data_df2.index) - 1:
                    data_df2.loc[j, 'B19013_001E'] = data_df.loc[i, 'B19013_001E']
                    j += 1
            data_df = data_df2
            data_df = data_df.fillna(0).replace(-666666666, 0)
            return data_df[self.attrib_vars['median_income']].to_numpy(dtype=np.int64).squeeze()
        return ADRIO('tract_median_income', fetch)

    def _make_commuter_adrio(self, granularity: int, nodes: dict[str, list[str]], year: int) -> ADRIO:
        """Makes an ADRIO to retrieve ACS commuting flow data."""
        def fetch() -> NDArray:
            data_df = self.fetch_commuters(granularity, nodes, year)
            # state level
            if granularity == Granularity.STATE.value:
                # get unique state identifier
                unique_states = ('0' + data_df['res_state_code']).unique()
                state_len = np.count_nonzero(unique_states)

                # create dictionary to be used as array indices
                states_enum = enumerate(unique_states)
                states_dict = dict((j, i) for i, j in states_enum)

                # group and aggregate data
                data_group = data_df.groupby(['res_state_code', 'wrk_state_code'])
                data_df = data_group.agg({'workers': 'sum'})

                # create and return array for each state
                output = np.zeros((state_len, state_len), dtype=np.int64)

                # fill array with commuting data
                for i, row in data_df.iterrows():
                    if type(i) is tuple:
                        x = states_dict.get('0' + i[0])
                        y = states_dict.get(i[1])

                        output[x][y] = row['workers']

            # county level
            else:
                # get unique identifier for each county
                geoid_df = DataFrame()
                geoid_df['geoid'] = '0' + data_df['res_state_code'] + \
                    data_df['res_county_code']
                unique_counties = geoid_df['geoid'].unique()

                # create empty output array
                county_len = np.count_nonzero(unique_counties)
                output = np.zeros((county_len, county_len), dtype=np.int64)

                # create dictionary to be used as array indices
                counties_enum = enumerate(unique_counties)
                counties_dict = dict((j, i) for i, j in counties_enum)

                data_df.reset_index(drop=True, inplace=True)

                # fill array with commuting data
                for i in range(len(data_df.index)):
                    x = counties_dict.get('0' +
                                          data_df.iloc[i]['res_state_code'] + data_df.iloc[i]['res_county_code'])
                    y = counties_dict.get(
                        data_df.iloc[i]['wrk_state_code'] + data_df.iloc[i]['wrk_county_code'])

                    output[x][y] = data_df.iloc[i]['workers']

            return output
        return ADRIO('commuters', fetch)

    def _make_simple_adrios(self, attrib: AttribDef, granularity: int, nodes: dict[str, list[str]], year: int) -> ADRIO:
        """Makes ADRIOs for simple attributes that require no additional postprocessing."""
        def fetch() -> NDArray:
            data_df = self.fetch_acs5(
                self.attrib_vars[attrib.name], granularity, nodes, year)
            if attrib.name == 'median_income' or attrib.name == 'median_age':
                data_df = data_df.fillna(0).replace(-666666666, 0)

            return data_df[self.attrib_vars[attrib.name]].to_numpy(dtype=attrib.dtype).squeeze()
        return ADRIO(attrib.name, fetch)
