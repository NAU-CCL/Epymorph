import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, concat

from epymorph.adrio.adrio import ADRIO


class DissimilarityIndex(ADRIO):
    """ADRIO to fetch the dissimilarity index of racial segregation for all counties in a provided set of states"""
    year = 2015
    attribute = 'dissimilarity_index'

    def __init__(self):
        super().__init__()

    def fetch(self, force=False, **kwargs) -> NDArray[np.float_]:
        """"Returns a numpy array of floats representing the disimilarity index for each county"""
        if force:
            uncached_tract = uncached_county = self.type_check(kwargs)
            cache_df_tract = cache_df_county = DataFrame()

        else:
            uncached_tract, cache_df_tract = self.cache_fetch(
                kwargs, '_tract')

            uncached_county, cache_df_county = self.cache_fetch(
                kwargs, '_county')

        code_string_tract = ','.join(uncached_tract)
        code_string_county = ','.join(uncached_county)

        if len(uncached_tract) > 0:
            # fetch tract level data from census
            tract_data = self.census.acs5.get(('B03002_003E',  # white population
                                               'B03002_013E',
                                               'B03002_004E',  # minority population
                                               'B03002_014E'),
                                              {'for': 'tract: *', 'in': 'county: *',
                                               'in': f'state: {code_string_tract}'},
                                              year=self.year)

            tract_data_df = DataFrame.from_records(tract_data)

            # sort data by state, county
            tract_data_df = tract_data_df.sort_values(by=['state', 'county'])
            tract_data_df.reset_index(inplace=True)

            self.cache_store(tract_data_df, uncached_tract, '_tract')
            if len(cache_df_tract.index) > 0:
                tract_data_df = concat([tract_data_df, cache_df_tract])

        else:
            cache_df_tract.reset_index(inplace=True)
            tract_data_df = cache_df_tract

        if len(uncached_county) > 0:
            # fetch county level data from census
            county_data = self.census.acs5.get(('B03002_003E',  # white population
                                                'B03002_013E',
                                                'B03002_004E',  # minority population
                                                'B03002_014E'),
                                               {'for': 'county: *',
                                                   'in': f'state: {code_string_county}'},
                                               year=self.year)

            county_data_df = DataFrame.from_records(county_data)

            # sort data by state, county
            county_data_df = county_data_df.sort_values(by=['state', 'county'])
            county_data_df.reset_index(inplace=True)

            self.cache_store(county_data_df, uncached_county, '_county')
            if len(cache_df_tract.index) > 0:
                county_data_df = concat([county_data_df, cache_df_county])

        else:
            cache_df_county.reset_index(inplace=True)
            county_data_df = cache_df_county

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
