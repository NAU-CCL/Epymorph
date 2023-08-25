import numpy as np
from numpy.typing import NDArray

from epymorph.adrio.census.adrio_census import ADRIO_census, Granularity


class DissimilarityIndex(ADRIO_census):
    """ADRIO to fetch the dissimilarity index of racial segregation"""
    attribute = 'dissimilarity_index'

    def fetch(self) -> NDArray[np.float_]:
        """"Returns a numpy array of floats representing the disimilarity index for each node"""
        # check for block group granularity and prepare to handle case
        cbg_granularity = False
        if self.granularity == Granularity.CBG.value:
            print("Dissimilarity index cannot be retrieved for block group level granularity, fetching tract level data instead")
            self.granularity = Granularity.TRACT.value
            cbg_granularity = True

        # fetch population data for requested granularity level from census
        upper_data_df = super().fetch(['B03002_003E',  # white population
                                       'B03002_013E',
                                       'B03002_004E',  # minority population
                                       'B03002_014E'])

        # fetch population data for one granularity down for use in calculation
        self.granularity += 1
        lower_data_df = super().fetch(['B03002_003E',  # white population
                                       'B03002_013E',
                                       'B03002_004E',  # minority population
                                       'B03002_014E'])
        self.granularity -= 1

        output = np.zeros(len(upper_data_df.index), dtype=np.float_)

        # loop for counties
        j = 0
        for i, county_iterator in upper_data_df.iterrows():
            # assign county fip to variable
            county_fip = county_iterator[str(
                Granularity(self.granularity).name).lower()]
            # loop for all tracts in county (while fip == variable)
            sum = 0.0
            while lower_data_df.iloc[j][str(Granularity(self.granularity).name).lower()] == county_fip and j < len(lower_data_df.index) - 1:
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

        # set cbg data to that of the parent tract if geo granularity = cbg
        if cbg_granularity:
            self.granularity = Granularity.CBG.value
            tract_output = output
            output = np.zeros(len(lower_data_df), dtype=np.float_)
            j = 0
            for i in range(len(upper_data_df.index)):
                tract_fip = upper_data_df.loc[i, 'tract']
                while lower_data_df.loc[j, 'tract'] == tract_fip and j < len(lower_data_df.index) - 1:
                    output[j] = tract_output[i]
                    j += 1

        return output
