import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from epymorph.adrio.adrio import ADRIO


class MedianAge(ADRIO):
    year = 2019
    attribute = 'median_age'

    def __init__(self) -> None:
        super().__init__()

    def fetch(self, force=False, **kwargs) -> NDArray[np.int_]:
        # skip caching and type checking for now - doing this for different distributions
        # with the current system is very awkward and would only lead to more case specific
        # logic that conflicts with what is established. This ADRIO will fetch data on its own until
        # a census template is made.

        state_string = county_string = ''
        county_list = kwargs.get('county')
        if type(county_list) is list:
            county_string = ','.join(county_list)
        state_list = kwargs.get('state')
        if type(state_list) is list:
            state_string = ','.join(state_list)

        # get data from census
        data = self.census.acs5.get('B01002_001E', {
                                    'for': 'block group: *',
                                    'in': f'state: {state_string} \
                                    county: {county_string}'}, year=self.year)

        data_df = DataFrame.from_records(data)

        # sort data by state and county fips
        data_df = data_df.sort_values(by=['state', 'county', 'block group'])
        data_df.reset_index(inplace=True)

        return data_df['B01002_001E'].to_numpy()
