import numpy as np
from numpy.typing import NDArray

from epymorph.adrio.adrio_census import ADRIO_census


class MedianAge(ADRIO_census):
    attribute = 'median_age'

    def fetch(self) -> NDArray[np.int_]:

        # get data from census
        data_df = super().fetch(['B01002_001E'])

        return data_df['B01002_001E'].to_numpy()
