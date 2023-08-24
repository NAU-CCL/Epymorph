import numpy as np
from numpy.typing import NDArray

from epymorph.adrio.adrio_census import ADRIO_census


class AverageHouseholdSize(ADRIO_census):
    attribute = 'average_household_size'

    def fetch(self) -> NDArray[np.int_]:

        # get data from census
        data_df = super().fetch(['B25010_001E'])

        return data_df['B25010_001E'].to_numpy()
