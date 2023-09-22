import numpy as np
from numpy.typing import NDArray

from epymorph.adrio.census.adrio_census import ADRIO_census, Granularity


class GiniIndex(ADRIO_census):
    """ADRIO to fetch the gini index of income inequality in a provided set of geographies"""
    attribute = 'gini_index'

    def fetch(self) -> NDArray[np.float64]:
        """Returns a numpy array of integers representing the gini index for each node"""
        # fetch tract level data if granularity = cbg
        cbg_df = None
        if self.granularity == Granularity.CBG.value:
            print(
                'Gini Index cannot be retrieved for block group level, fetching tract level data instead.')
            # this actually gets cbg data, but it is dummy data to be overwritten later
            cbg_df = super().fetch(['B19083_001E'])
            self.granularity = Granularity.TRACT.value

        # get data from census
        data_df = super().fetch(['B19083_001E'])

        data_df['B19083_001E'] = data_df['B19083_001E'].astype(
            np.float64).fillna(0.5).replace(-666666666, 0.5)

        # set cbg data to that of the parent tract if geo granularity = cbg
        if self.granularity == Granularity.TRACT.value and cbg_df is not None:
            j = 0
            for i in range(len(data_df.index)):
                tract_fip = data_df.loc[i, 'tract']
                while cbg_df.loc[j, 'tract'] == tract_fip and j < len(cbg_df.index) - 1:
                    cbg_df.loc[j, 'B19083_001E'] = data_df.loc[i, 'B19083_001E']
                    j += 1
            data_df = cbg_df

        return data_df['B19083_001E'].to_numpy(dtype=np.float64)
