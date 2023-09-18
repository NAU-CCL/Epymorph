import os
from enum import Enum

from census import Census
from geopandas import GeoDataFrame
from pandas import DataFrame, read_excel
from pygris import block_groups, counties, states, tracts

from epymorph.adrio import ADRIO
from epymorph.adrio.adrio import GEOSpec


class Granularity(Enum):
    STATE = 0
    COUNTY = 1
    TRACT = 2
    CBG = 3


class ADRIO_census(ADRIO):
    """
    Census ADRIO template to serve as parent class and provide utility functions for Census-based ADRIOS
    """
    census: Census
    year: int
    granularity: int
    nodes: dict[str, list[str]]

    def __init__(self, **kwargs) -> None:
        """
        Initializer to create Census object and set query properties
        """
        spec = kwargs.get('spec')
        if type(spec) is GEOSpec:
            self.year = spec.year
            self.granularity = spec.granularity
            self.nodes = spec.nodes
        else:
            msg = 'One or more parameters are missing or formatted incorrectly'
            raise Exception(msg)

        self.census = Census(os.environ['CENSUS_API_KEY'])

    def fetch(self, variables: list[str]) -> DataFrame:
        """
        Utility function to fetch Census data by building queries from ADRIO data
        """
        # verify node types and convert to strings usable by census api
        states = self.nodes.get('state')
        counties = self.nodes.get('county')
        tracts = self.nodes.get('tract')
        cbg = self.nodes.get('block group')
        if type(states) is list:
            states = ','.join(states)
        if type(counties) is list:
            counties = ','.join(counties)
        if type(tracts) is list:
            tracts = ','.join(tracts)
        if type(cbg) is list:
            cbg = ','.join(cbg)

        # fetch and sort data according to granularity
        if self.granularity == Granularity.STATE.value:
            data = self.census.acs5.get(
                variables, {'for': f'state: {states}'}, year=self.year)
            data_df = DataFrame.from_records(data)
            data_df = data_df.sort_values(by=['state'])
        elif self.granularity == Granularity.COUNTY.value:
            data = self.census.acs5.get(
                variables, {'for': f'county: {counties}', 'in': f'state: {states}'}, year=self.year)
            data_df = DataFrame.from_records(data)
            data_df = data_df.sort_values(by=['state', 'county'])
        elif self.granularity == Granularity.TRACT.value:
            data = self.census.acs5.get(variables, {
                                        'for': f'tract: {tracts}', 'in': f'state: {states} county: {counties}'}, year=self.year)
            data_df = DataFrame.from_records(data)
            data_df = data_df.sort_values(by=['state', 'county', 'tract'])
        else:
            data = self.census.acs5.get(variables, {
                                        'for': f'block group: {cbg}', 'in': f'state: {states} county: {counties} tract: {tracts}'}, year=self.year)
            data_df = DataFrame.from_records(data)
            data_df = data_df.sort_values(
                by=['state', 'county', 'tract', 'block group'])

        data_df.reset_index(inplace=True)

        # return data to adrio for processing
        return data_df

    def fetch_sf(self) -> GeoDataFrame:
        """
        Utility function to fetch shape files from Census for specified regions
        """
        # call appropriate pygris function based on granularity and sort result
        if self.granularity == Granularity.STATE.value:
            data_df = states(year=self.year)
            data_df = data_df.rename(columns={'STATEFP': 'state'})

        elif self.granularity == Granularity.COUNTY.value:
            data_df = counties(state=self.nodes.get('state'), year=self.year)
            data_df = data_df.rename(columns={'STATEFP': 'state', 'COUNTYFP': 'county'})
            data_df = data_df.sort_values(by=['state', 'county'])
            data_df.reset_index(drop=True, inplace=True)

        elif self.granularity == Granularity.TRACT.value:
            state_fips = self.nodes.get('state')
            county_fips = self.nodes.get('county')
            if state_fips is not None and county_fips is not None:
                data_df = GeoDataFrame()
                # tract and block group level files cannot be fetched using lists
                # several queries must be made and merged instead
                for i in range(len(state_fips)):
                    for j in range(len(county_fips)):
                        current_data = tracts(
                            state=state_fips[i], county=county_fips[j], year=self.year)

                        if len(data_df.index) > 0:
                            data_df = data_df.merge(
                                current_data, on=['STATEFP', 'COUNTYFP', 'TRACTCE'])
                        else:
                            data_df = current_data
            else:
                msg = "Data could not be retrieved due to missing state, county fips codes"
                raise Exception(msg)

            data_df = data_df.rename(
                columns={'STATEFP': 'state', 'COUNTYFP': 'county', 'TRACTCE': 'tract'})
            data_df = GeoDataFrame(data_df.sort_values(by=['state', 'county', 'tract']))
            data_df.reset_index(drop=True, inplace=True)

        else:
            state_fips = self.nodes.get('state')
            county_fips = self.nodes.get('county')
            if state_fips is not None and county_fips is not None:
                data_df = GeoDataFrame()
                for i in range(len(state_fips)):
                    for j in range(len(county_fips)):
                        current_data = block_groups(
                            state=state_fips[i], county=county_fips[j], year=self.year)
                        if len(data_df.index) > 0:
                            data_df = data_df.merge(
                                current_data, on=['STATEFP', 'COUNTYFP', 'TRACTCE', 'BLKGRPCE'])
                        else:
                            data_df = current_data
            else:
                msg = "Data could not be retrieved due to missing state or county fips codes"
                raise Exception(msg)

            data_df = data_df.rename(
                columns={'STATEFP': 'state', 'COUNTYFP': 'county', 'TRACTCE': 'tract', 'BLKGRPCE': 'block group'})
            data_df = GeoDataFrame(data_df.sort_values(
                by=['state', 'county', 'block group']))
            data_df.reset_index(drop=True, inplace=True)

        return data_df

    def fetch_commuters(self) -> DataFrame:
        """
        Utility function to fetch commuting data from .xslx format filtered down to requested regions
        """
        # check for valid year
        if self.year not in [2010, 2015, 2020]:
            # if invalid year is close to a valid year, fetch valid data and notify user
            passed_year = self.year
            if self.year in range(2008, 2012):
                self.year = 2010
            elif self.year in range(2013, 2017):
                self.year = 2015
            elif self.year in range(2018, 2022):
                self.year = 2020
            else:
                msg = "Invalid year. Communting data is only available for 2008-2022"
                raise Exception(msg)

            print(
                f"Commuting data cannot be retrieved for {passed_year}, fetching {self.year} data instead.")

        # check for invalid granularity
        if self.granularity == Granularity.CBG.value or self.granularity == Granularity.TRACT.value:
            print(
                'Commuting data cannot be retrieved for tract or block group granularities,', end='')
            print('fetching county level data instead.')
            self.granularity = Granularity.COUNTY.value

        states = self.nodes.get('state')
        counties = self.nodes.get('county')

        if self.year != 2010:
            url = f'https://www2.census.gov/programs-surveys/demo/tables/metro-micro/{self.year}/commuting-flows-{self.year}/table1.xlsx'

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

        if self.granularity == Granularity.COUNTY.value:
            if counties is not None and counties[0] != '*':
                data = data.loc[data['res_county_code'].isin(counties)]
                data = data.loc[data['wrk_county_code'].isin(counties)]

        return data
