import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, concat
from pygris import counties

from epymorph.adrio.adrio import ADRIO


class PopDensityKm2(ADRIO):
    """ADRIO to fetch population density for each county in a provided set of states"""
    year = 2015
    attribute = 'pop_density_km2'

    def __init__(self):
        super().__init__()

    def fetch(self, force=False, **kwargs) -> NDArray[np.float_]:
        """Returns a numpy array of floats representing the population density for each county"""
        if force:
            uncached_pop = uncached_geo = self.type_check(kwargs)
            cache_pop_df = cache_geo_df = DataFrame()
        else:
            uncached_pop, cache_pop_df = self.cache_fetch(kwargs, '_pop')
            uncached_geo, cache_geo_df = self.cache_fetch(kwargs, '_geo')

        code_string_pop = ','.join(uncached_pop)

        # get county shapefiles
        if len(uncached_geo) > 0:
            county_df = counties(state=uncached_geo, year=self.year)

            county_df = county_df.rename(
                columns={'STATEFP': 'state', 'COUNTYFP': 'county'})

            # sort data by state and county fips
            county_df = county_df.sort_values(by=['state', 'county'])
            county_df.reset_index(drop=True, inplace=True)

            self.cache_store(county_df, uncached_geo, '_geo')
            if len(cache_geo_df.index) > 0:
                county_df = concat([county_df, cache_geo_df])

        else:
            county_df = cache_geo_df

        if len(uncached_pop) > 0:
            # fetch county population data from census
            census_data = self.census.acs5.get(
                'B01003_001E', {'for': 'county: *', 'in': f'state: {code_string_pop}'}, year=self.year)

            census_df = DataFrame.from_records(census_data)

            # sort data by state and county fips
            census_df = census_df.sort_values(by=['state', 'county'])
            census_df.reset_index(drop=True, inplace=True)

            self.cache_store(census_df, uncached_pop, '_pop')
            if len(cache_pop_df.index) > 0:
                census_df = concat([census_df, cache_pop_df])

        else:
            census_df = cache_pop_df

        # merge census data with shapefile data
        county_df = county_df.merge(census_df, on=['state', 'county'])

        # calculate population density, storing it in a numpy array to return
        output = np.zeros(len(county_df.index), dtype=np.float_)
        for i, row in county_df.iterrows():
            output[i] = round(int(row['B01003_001E']) / (row['ALAND'] / 1e6))
        return output
