import numpy as np
from numpy.typing import NDArray

from epymorph.adrio.census.adrio_census import ADRIO_census


class AverageHouseholdSize(ADRIO_census):
    """ADRIO to fetch the average household size for a provided set of geographies"""
    attribute = 'average_household_size'

    def fetch(self) -> NDArray[np.int_]:
        """Returns a numpy array of integers representing the average household size in each node"""
        # get data from census
        data_df = super().fetch(['B25010_001E'])

        return data_df['B25010_001E'].to_numpy(dtype=np.int_)
