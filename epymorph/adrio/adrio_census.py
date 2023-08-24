import os
from enum import Enum

from census import Census
from geopandas import GeoDataFrame
from pandas import DataFrame
from pygris import block_groups, counties, states, tracts

from epymorph.adrio import ADRIO
from epymorph.adrio.adrio import GEOSpec


class Granularity(Enum):
    STATE = 0
    COUNTY = 1
    TRACT = 2
    CBG = 3


class ADRIO_census(ADRIO):
    """abstract class to serve as an outline for individual ADRIO implementations"""
    census: Census
    year: int
    granularity: int
    nodes: dict[str, list[str]]

    def __init__(self, **kwargs):
        """
        initializer to create Census object
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
                                        'for': f'block group: {cbg}', 'in': f'state: {states} county: {counties}'}, year=self.year)
            data_df = DataFrame.from_records(data)
            data_df = data_df.sort_values(by=['state', 'county', 'block group'])

        data_df.reset_index(inplace=True)

        # return data to adrio for processing
        return data_df

    def fetch_sf(self) -> GeoDataFrame:
        if self.granularity == Granularity.STATE.value:
            data_df = states(year=self.year)
            data_df = data_df.rename(columns={'STATEFP': 'state'})
        elif self.granularity == Granularity.COUNTY.value:
            data_df = counties(state=self.nodes.get('state'), year=self.year)

            data_df = data_df.rename(
                columns={'STATEFP': 'state', 'COUNTYFP': 'county'})

            # retreive counties from states of interest and sort by fips
            data_df = data_df.sort_values(by=['state', 'county'])
            data_df.reset_index(drop=True, inplace=True)
        elif self.granularity == Granularity.TRACT.value:
            data_df = tracts(state=self.nodes.get('state'),
                             county=self.nodes.get('county'), year=self.year)

            data_df = data_df.rename(
                columns={'STATEFP': 'state', 'COUNTYFP': 'county', 'TRACTFP': 'tract'})

            # retreive counties from states of interest and sort by fips
            data_df = data_df.sort_values(by=['state', 'county', 'tract'])
            data_df.reset_index(drop=True, inplace=True)
        else:
            data_df = block_groups(state=self.nodes.get(
                'state'), county=self.nodes.get('county'), year=self.year)

            data_df = data_df.rename(
                columns={'STATEFP': 'state', 'COUNTYFP': 'county', 'BLOCKGROUPFP': 'block group'})

            # retreive counties from states of interest and sort by fips
            data_df = data_df.sort_values(by=['state', 'county', 'block group'])
            data_df.reset_index(drop=True, inplace=True)

        return data_df
