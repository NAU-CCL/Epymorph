import numpy as np
from numpy.typing import NDArray

from epymorph.adrio.adrio_census import ADRIO_census


class NameAndState(ADRIO_census):
    """ADRIO to fetch the names and state names of every county in a provided set of states"""
    attribute = 'name_and_state'

    def fetch(self) -> NDArray[np.str_]:
        """Returns a numpy array of containing the name and state name of each county as strings"""
        data_df = super().fetch(['NAME'])

        # store county and state names in numpy array to return
        return data_df['NAME'].to_numpy()
