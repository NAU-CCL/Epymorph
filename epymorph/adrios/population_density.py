import geopandas as gpd
import numpy as np
import pandas as pd
from census import Census
from numpy.typing import NDArray

from epymorph.adrio import ADRIO


class PopulationDensity(ADRIO):
    """
    ADRIO to fetch population density for each county in a provided set of states
    Returns an array of floats representing the population density for each county
    """
    census: Census
    year = 2015
    attribute = 'population density'

    def __init__(self, key: str):
        self.census = Census(key)

    def fetch(self, **kwargs) -> NDArray[np.float_]:
        geo_codes = kwargs.get('nodes')
        code_string = self.format_geo_codes(kwargs)

        # get county shape file for the specified year
        url = f'https://www2.census.gov/geo/tiger/GENZ{self.year}/shp/cb_{self.year}_us_county_500k.zip'
        all_counties = gpd.read_file(url)

        # retreive counties from states of interest
        counties = all_counties.loc[all_counties['STATEFP'].isin(
            geo_codes)]

        data = self.census.acs5.get(('NAME', 'B01003_001E'), {
                                    'for': 'county: *', 'in': 'state: {}'.format(code_string)}, year=self.year)

        # merge census data with shapefile data
        census_df = pd.DataFrame.from_records(data)
        census_df = census_df.rename(
            columns={'county': 'COUNTYFP', 'B01003_001E': 'POPULATION', 'state': 'STATEFP'})
        counties = counties.drop(columns=['NAME', 'STATEFP'])
        counties = counties.merge(census_df, on='COUNTYFP')

        # sort data by state and county fips
        counties = counties.sort_values(by=['STATEFP', 'COUNTYFP'])

        # calculate population density, storing it in a numpy array to return
        output = np.zeros((len(counties.index),), dtype=np.float_)
        i = 0
        for index, row in counties.iterrows():
            output[i] = round(
                row['POPULATION'] / row['geometry'].area, 2)
            i += 1
        return output
