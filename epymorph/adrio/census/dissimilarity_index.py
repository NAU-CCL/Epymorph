import numpy as np
from numpy.typing import NDArray

from epymorph.adrio.adrio_census import ADRIO_census


class DissimilarityIndex(ADRIO_census):
    """ADRIO to fetch the dissimilarity index of racial segregation for all counties in a provided set of states"""
    attribute = 'dissimilarity_index'

    def fetch(self) -> NDArray[np.float_]:
        """"Returns a numpy array of floats representing the disimilarity index for each county"""

        # fetch county level data from census
        upper_data_df = super().fetch(['B03002_003E',  # white population
                                       'B03002_013E',
                                       'B03002_004E',  # minority population
                                       'B03002_014E'])

        self.granularity += 1

        # fetch tract level data from census
        lower_data_df = super().fetch(['B03002_003E',  # white population
                                       'B03002_013E',
                                       'B03002_004E',  # minority population
                                       'B03002_014E'])

        output = np.zeros(len(upper_data_df.index), dtype=np.float_)

        # loop for counties
        j = 0
        for i, county_iterator in upper_data_df.iterrows():
            # assign county fip to variable
            county_fip = county_iterator['county']
            # loop for all tracts in county (while fip == variable)
            sum = 0.0
            while lower_data_df.iloc[j]['county'] == county_fip and j < len(lower_data_df.index) - 1:
                # preliminary calculations
                tract_minority = lower_data_df.iloc[j]['B03002_004E'] + \
                    lower_data_df.iloc[j]['B03002_014E']
                county_minority = county_iterator['B03002_004E'] + \
                    county_iterator['B03002_014E']
                tract_majority = lower_data_df.iloc[j]['B03002_003E'] + \
                    lower_data_df.iloc[j]['B03002_013E']
                county_majority = county_iterator['B03002_003E'] + \
                    county_iterator['B03002_013E']

                # run calculation sum += ( |minority(tract) / minority(county) - majority(tract) / majority(county)| )
                if county_minority != 0 and county_majority != 0:
                    sum = sum + abs(tract_minority / county_minority -
                                    tract_majority / county_majority)
                j += 1

            sum *= .5
            if sum == 0.:
                sum = 0.5

            # assign current output element to sum
            output[i] = sum

        return output
