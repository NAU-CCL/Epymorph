import numpy as np
from numpy.typing import NDArray
from pygris import counties

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

        # get county shapefile for the specified year
        county_df = counties(state=geo_codes, year=self.year, cache=True)

        # retreive counties from states of interest and sort by fips
        county_df = county_df.sort_values(by=['STATEFP', 'COUNTYFP'])
        county_df.reset_index(drop=True, inplace=True)

        # map county's name to its centroid in a numpy array and return
        output = np.zeros(len(county_df.index), dtype=tuple)
        for i, row in county_df.iterrows():
            output[i] = row['geometry'].centroid.coords[0]

        return output
