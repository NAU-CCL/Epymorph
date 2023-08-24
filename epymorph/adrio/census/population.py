import numpy as np
from numpy.typing import NDArray

from epymorph.adrio.adrio_census import ADRIO_census


class Population(ADRIO_census):
    """ADRIO to fetch total population in all counties in a provided set of states"""
    attribute = 'population'

    def fetch(self) -> NDArray[np.int_]:
        """Returns a numpy array containing each county's total population"""

        data_df = super().fetch(['B01001_001E'])

        # convert to numy array and return
        return data_df['B01001_001E'].to_numpy(dtype=np.int_)
