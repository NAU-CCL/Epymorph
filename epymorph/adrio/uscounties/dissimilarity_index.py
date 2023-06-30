import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from epymorph.adrio.adrio import ADRIO


class DissimilarityIndex(ADRIO):
    """ADRIO to fetch the dissimilarity index of racial segregation for all counties in a provided set of states"""
    year = 2015
    attribute = 'dissimilarity index'

    def __init__(self):
        super().__init__()

    def fetch(self, **kwargs) -> NDArray[np.float_]:
        """"Returns a numpy array of floats representing the disimilarity index for each county"""
        code_string = self.type_check(kwargs)
        code_string = ','.join(code_string)

        # fetch tract level data from census
        tract_data = self.census.acs5.get(('B03002_003E',  # white population
                                           'B03002_013E',
                                           'B03002_004E',  # minority population
                                           'B03002_014E'),
                                          {'for': 'tract: *', 'in': 'county: *',
                                           'in': f'state: {code_string}'},
                                          year=self.year)

        # fetch county level data from census
        county_data = self.census.acs5.get(('B03002_003E',  # white population
                                            'B03002_013E',
                                            'B03002_004E',  # minority population
                                            'B03002_014E'),
                                           {'for': 'county: *',
                                               'in': f'state: {code_string}'},
                                           year=self.year)

        # sort data by state, county
        tract_data_df = DataFrame.from_records(tract_data)
        tract_data_df = tract_data_df.sort_values(by=['state', 'county'])
        tract_data_df.reset_index(inplace=True)

        county_data_df = DataFrame.from_records(county_data)
        county_data_df = county_data_df.sort_values(by=['state', 'county'])
        county_data_df.reset_index(inplace=True)

        output = np.zeros(len(county_data_df.index), dtype=np.float_)

        # loop for counties
        j = 0
        for i, county_iterator in county_data_df.iterrows():
            # assign county fip to variable
            county_fip = county_iterator['county']
            # loop for all tracts in county (while fip == variable)
            sum = 0.0
            while tract_data_df.iloc[j]['county'] == county_fip and j < len(tract_data_df.index) - 1:
                # preliminary calculations
                tract_minority = tract_data_df.iloc[j]['B03002_004E'] + \
                    tract_data_df.iloc[j]['B03002_014E']
                county_minority = county_iterator['B03002_004E'] + \
                    county_iterator['B03002_014E']
                tract_majority = tract_data_df.iloc[j]['B03002_003E'] + \
                    tract_data_df.iloc[j]['B03002_013E']
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
