import numpy as np
from numpy.typing import NDArray

from epymorph.adrio.census.adrio_census import ADRIO_census


class MedianAge(ADRIO_census):
    """ADRIO to fetch the median age for a provided set of geographies"""
    attribute = 'median_age'

    def fetch(self) -> NDArray[np.float64]:
        """Returns a numpy array of integers representing the median age in each node"""
        # get data from census
        data_df = super().fetch(['B01002_001E'])

        data_df.fillna(0).replace(-666666666, 0.0)

        return data_df['B01002_001E'].to_numpy(np.float64)
