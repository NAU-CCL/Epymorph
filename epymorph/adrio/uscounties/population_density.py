import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pygris import counties

from epymorph.adrio.adrio import ADRIO


class PopulationDensity(ADRIO):
    """ADRIO to fetch population density for each county in a provided set of states"""
    year = 2015
    attribute = 'population density'

    def __init__(self):
        super().__init__()

    def fetch(self, **kwargs) -> NDArray[np.float_]:
        """Returns a numpy array of floats representing the population density for each county"""
        geo_codes = self.type_check(kwargs)
        code_string = ','.join(geo_codes)

        # get county shapefiles
        county_df = counties(state=geo_codes, year=self.year, cache=True)

        # fetch county population data from census
        census_data = self.census.acs5.get(
            'B01003_001E', {'for': 'county: *', 'in': f'state: {code_string}'}, year=self.year)

        census_df = pd.DataFrame.from_records(census_data)

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
