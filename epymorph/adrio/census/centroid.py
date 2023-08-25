import numpy as np
from numpy.typing import NDArray

from epymorph.adrio.census.adrio_census import ADRIO_census

CentroidDType = np.dtype([('longitude', float), ('latitude', float)])


class Centroid(ADRIO_census):
    """ADRIO to fetch the geographic centroid of all provided geographies"""
    attribute = 'centroid'

    def fetch(self) -> NDArray:
        """
        Returns a numpy array of tuples representing an ordered pair of coordinates for each
        nodes's geographic centroid
        """

        data_df = super().fetch_sf()

        # map node's name to its centroid in a numpy array and return
        output = np.zeros(len(data_df.index), dtype=CentroidDType)
        for i in range(len(data_df.index)):
            output[i] = data_df.iloc[i]['geometry'].centroid.coords[0]

        return output
