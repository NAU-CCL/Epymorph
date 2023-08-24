import numpy as np
from numpy.typing import NDArray
from shapely.wkt import loads

from epymorph.adrio.adrio_census import ADRIO_census

CentroidDType = np.dtype([('longitude', float), ('latitude', float)])


class Centroid(ADRIO_census):
    """ADRIO to fetch the geographic centroid of all counties in a provided set of states"""
    attribute = 'centroid'

    def fetch(self) -> NDArray:
        """
        Returns a numpy array of tuples that are each an ordered pair of coordinates for each
        county's geographic centroid
        """

        county_df = super().fetch_sf()

        # map county's name to its centroid in a numpy array and return
        output = np.zeros(len(county_df.index), dtype=CentroidDType)
        for i, row in county_df.iterrows():
            if type(row['geometry']) is str:
                output[i] = loads(row['geometry']).centroid.coords[0]
            else:
                output[i] = row['geometry'].centroid.coords[0]

        return output
