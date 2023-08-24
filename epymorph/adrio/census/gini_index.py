import numpy as np
from numpy.typing import NDArray

from epymorph.adrio.adrio_census import ADRIO_census


class GiniIndex(ADRIO_census):
    attribute = 'gini_index'

    def fetch(self) -> NDArray[np.float_]:

        # get data from census
        data_df = super().fetch(['B19083_001E'])

        data_df['B19083_001E'] = data_df['B19083_001E'].astype(
            float).fillna(0.5).replace(-666666666, 0.5)

        return data_df['B19083_001E'].to_numpy()
