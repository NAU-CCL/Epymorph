import numpy as np
from geopandas import GeoDataFrame
from numpy.typing import NDArray
from pandas import concat
from pygris import counties
from shapely.wkt import loads

from epymorph.adrio.adrio import ADRIO


class Centroid(ADRIO):
    """ADRIO to fetch the geographic centroid of all counties in a provided set of states"""
    attribute = 'centroid'
    year = 2015

    def fetch(self, force=False, **kwargs) -> NDArray[np.float_]:
        """
        Returns a numpy array of tuples that are each an ordered pair of coordinates for each
        county's geographic centroid
        """
        if force:
            uncached = self.type_check(kwargs)
            cache_df = GeoDataFrame()
        else:
            uncached, cache_df = self.cache_fetch(kwargs)

        if len(uncached) > 0:
            # get county shapefile for the specified year
            county_df = counties(state=uncached, year=self.year)

            county_df = county_df.rename(
                columns={'STATEFP': 'state', 'COUNTYFP': 'county'})

            # retreive counties from states of interest and sort by fips
            county_df = county_df.sort_values(by=['state', 'county'])
            county_df.reset_index(drop=True, inplace=True)

            self.cache_store(county_df, uncached)
            if len(cache_df.index) > 0:
                county_df = concat([county_df, cache_df])

        else:
            county_df = cache_df

        # map county's name to its centroid in a numpy array and return
        output = np.zeros(len(county_df.index), dtype=tuple)
        for i, row in county_df.iterrows():
            if type(row['geometry']) is str:
                output[i] = loads(row['geometry']).centroid.coords[0]
            else:
                output[i] = row['geometry'].centroid.coords[0]

        return output
