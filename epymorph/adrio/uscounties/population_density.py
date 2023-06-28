import geopandas as gpd
import numpy as np
import pandas as pd
from numpy.typing import NDArray

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

        # get county shape file for the specified year
        url = f'https://www2.census.gov/geo/tiger/GENZ{self.year}/shp/cb_{self.year}_us_county_500k.zip'
        all_counties = gpd.read_file(url)

        # retreive counties from states of interest
        counties = all_counties.loc[all_counties['STATEFP'].isin(
            geo_codes)]

        # fetch county population data from census
        census_data = self.census.acs5.get(
            'B01003_001E', {'for': 'county: *', 'in': f'state: {code_string}'}, year=self.year)

        census_df = pd.DataFrame.from_records(census_data)

        # merge census data with shapefile data
        census_df = census_df.rename(
            columns={'county': 'COUNTYFP', 'B01003_001E': 'POPULATION', 'state': 'STATEFP'})

        counties = counties.drop(columns=['STATEFP'])
        counties = counties.merge(census_df, on='COUNTYFP')

        # sort data by state and county fips
        counties = counties.sort_values(by=['STATEFP', 'COUNTYFP'])
        counties.reset_index(drop=True, inplace=True)

        # calculate population density, storing it in a numpy array to return
        output = np.zeros(len(counties.index), dtype=np.float_)
        for i, row in counties.iterrows():
            output[i] = round(
                int(row['POPULATION']) / row['geometry'].area, 2)
        return output
