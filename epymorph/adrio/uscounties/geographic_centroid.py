import geopandas as gpd
import numpy as np
from numpy.typing import NDArray

from epymorph.adrio.adrio import ADRIO


class GeographicCentroid(ADRIO):
    """ADRIO to fetch the geographic centroid of all counties in a provided set of states"""
    attribute = 'geographic centroid'
    year = 2015

    def fetch(self, **kwargs) -> NDArray[np.float_]:
        """
        Returns a numpy array of tuples that are each an ordered pair of coordinates for each
        county's geographic centroid
        """
        geo_codes = kwargs.get("nodes")

        # get county shape file for the specified year
        url = f'https://www2.census.gov/geo/tiger/GENZ{self.year}/shp/cb_{self.year}_us_county_500k.zip'
        all_counties = gpd.read_file(url)

        # retreive counties from states of interest and sort by fips
        counties = all_counties.loc[all_counties['STATEFP'].isin(
            geo_codes)]
        counties = counties.sort_values(by=['STATEFP', 'COUNTYFP'])
        counties.reset_index(drop=True, inplace=True)

        # map county's name to its centroid in a numpy array and return
        output = np.zeros(len(counties.index), dtype=tuple)
        for i, row in counties.iterrows():
            output[i] = row['geometry'].centroid.coords[0]

        return output
