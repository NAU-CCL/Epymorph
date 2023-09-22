import numpy as np
from numpy.typing import NDArray

from epymorph.adrio.census.adrio_census import ADRIO_census


class Name(ADRIO_census):
    """ADRIO to fetch the names of each node provided"""
    attribute = 'name'

    def fetch(self) -> NDArray[np.str_]:
        """Returns a numpy array of containing the name of each geographic node"""
        data_df = super().fetch(['NAME'])

        # store county and state names in numpy array to return
        return data_df['NAME'].to_numpy(dtype=np.str_)
