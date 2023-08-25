import numpy as np
from numpy.typing import NDArray

from epymorph.adrio.census.adrio_census import ADRIO_census, Granularity


class TractMedianIncome(ADRIO_census):
    """Case specific ADRIO used to retreive tract level median income data for geos with block group granularity"""
    attribute = 'tract_median_income'

    def fetch(self) -> NDArray[np.int_]:
        """
        Returns a numpy array of integers representing the tract level median annual 
        household income for each census block group's parent tract
        """

        # fetch cbg data from census
        cbg_df = super().fetch(['B19013_001E'])

        self.granularity = Granularity.TRACT.value

        # fetch tract data from census
        tract_df = super().fetch(['B19013_001E'])

        tract_df['B19013_001E'] = tract_df['B19013_001E'].astype(
            float).fillna(0.5).replace(-666666666, 0.5)

        # set cbg data to that of the parent tract
        j = 0
        for i in range(len(tract_df.index)):
            tract_fip = tract_df.loc[i, 'tract']
            while cbg_df.loc[j, 'tract'] == tract_fip and j < len(cbg_df.index) - 1:
                cbg_df.loc[j, 'B19013_001E'] = tract_df.loc[i, 'B19013_001E']
                j += 1
        tract_df = cbg_df

        # convert to numpy array and return
        return tract_df['B19013_001E'].to_numpy(dtype=np.int_)
