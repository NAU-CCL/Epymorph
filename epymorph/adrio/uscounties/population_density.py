import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, concat
from pygris import counties

from epymorph.adrio.adrio import ADRIO


class PopulationDensity(ADRIO):
    """ADRIO to fetch population density for each county in a provided set of states"""
    year = 2015
    attribute = 'population density'

    def __init__(self):
        super().__init__()

    def fetch(self, force=False, **kwargs) -> NDArray[np.float_]:
        """Returns a numpy array of floats representing the population density for each county"""
        geo_codes = self.type_check(kwargs)

        if force:
            code_list = geo_codes
            cache_df = DataFrame()
        else:
            cache_data = self.cache_fetch(kwargs, self.attribute)
            code_list = cache_data[0]
            cache_df = cache_data[1]

        code_string = ','.join(code_list)

        # get county shapefiles
        county_df = counties(state=geo_codes, year=self.year, cache=True)

        if len(code_list) > 0:
            # fetch county population data from census
            census_data = self.census.acs5.get(
                'B01003_001E', {'for': 'county: *', 'in': f'state: {code_string}'}, year=self.year)

            census_df = DataFrame.from_records(census_data)
            self.cache_store(census_df, code_list, self.attribute)
            if len(cache_df.index) > 0:
                census_df = concat([census_df, cache_df])

        else:
            census_df = cache_df

        county_df['COUNTYFP'] = county_df['COUNTYFP'].astype(int)
        census_df['county'] = census_df['county'].astype(int)

        # merge census data with shapefile data
        census_df = census_df.rename(
            columns={'county': 'COUNTYFP', 'B01003_001E': 'POPULATION', 'state': 'STATEFP'})

        county_df = county_df.drop(columns=['STATEFP'])
        county_df = county_df.merge(census_df, on='COUNTYFP')

        # sort data by state and county fips
        county_df = county_df.sort_values(by=['STATEFP', 'COUNTYFP'])
        county_df.reset_index(drop=True, inplace=True)

        # calculate population density, storing it in a numpy array to return
        output = np.zeros(len(county_df.index), dtype=np.float_)
        for i, row in county_df.iterrows():
            output[i] = round(
                int(row['POPULATION']) / row['geometry'].area, 2)
        return output
