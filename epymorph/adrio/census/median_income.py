import numpy as np
from numpy.typing import NDArray

from epymorph.adrio.census.adrio_census import ADRIO_census


class MedianIncome(ADRIO_census):
    """ADRIO to fetch median household income for a provided set of states"""
    attribute = 'median_income'

    def fetch(self) -> NDArray[np.int64]:
        """Returns a numpy array of integers representing the median annual household income for each node"""
        # fetch data from census
        data_df = super().fetch(['B19013_001E'])

        data_df = data_df.fillna(0).replace(-666666666, 0)

        # convert to numpy array and return
        return data_df['B19013_001E'].to_numpy(dtype=np.int64)
