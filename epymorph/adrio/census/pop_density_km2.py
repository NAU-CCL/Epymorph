import numpy as np
from numpy.typing import NDArray

from epymorph.adrio.census.adrio_census import ADRIO_census, Granularity


class PopDensityKm2(ADRIO_census):
    """ADRIO to fetch population density for a provided set of geographies"""
    attribute = 'pop_density_km2'

    def fetch(self) -> NDArray[np.float64]:
        """Returns a numpy array of floats representing the population density for each node"""

        # get shapefiles
        geo_df = super().fetch_sf()

        # fetch population data from census
        census_df = super().fetch(['B01003_001E'])

        # merge census data with shapefile data
        if self.granularity == Granularity.STATE.value:
            geo_df = geo_df.merge(census_df, on=['state'])
        elif self.granularity == Granularity.COUNTY.value:
            geo_df = geo_df.merge(census_df, on=['state', 'county'])
        elif self.granularity == Granularity.TRACT.value:
            geo_df = geo_df.merge(census_df, on=['state', 'county', 'tract'])
        else:
            geo_df = geo_df.merge(
                census_df, on=['state', 'county', 'tract', 'block group'])

        # calculate population density, storing it in a numpy array to return
        output = np.zeros(len(geo_df.index), dtype=np.float64)
        for i, row in geo_df.iterrows():
            output[i] = round(int(row['B01003_001E']) / (row['ALAND'] / 1e6))
        return output
