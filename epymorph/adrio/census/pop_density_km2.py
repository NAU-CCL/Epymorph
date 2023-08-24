import numpy as np
from numpy.typing import NDArray
from pygris import counties

from epymorph.adrio.adrio_census import ADRIO_census


class PopDensityKm2(ADRIO_census):
    """ADRIO to fetch population density for each county in a provided set of states"""
    attribute = 'pop_density_km2'

    def fetch(self) -> NDArray[np.float_]:
        """Returns a numpy array of floats representing the population density for each county"""

        # get county shapefiles
        county_df = super().fetch_sf()

        # fetch county population data from census
        census_df = super().fetch(['B01003_001E'])

        # merge census data with shapefile data
        county_df = county_df.merge(census_df, on=['state', 'county'])

        # calculate population density, storing it in a numpy array to return
        output = np.zeros(len(county_df.index), dtype=np.float_)
        for i, row in county_df.iterrows():
            output[i] = round(int(row['B01003_001E']) / (row['ALAND'] / 1e6))
        return output
