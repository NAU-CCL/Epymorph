import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, concat

from epymorph.adrio.adrio import ADRIO


class MedianIncome(ADRIO):
    """ADRIO to fetch median household income for each county in a provided set of states"""
    year = 2015
    attribute = 'median_income'

    def __init__(self) -> None:
        super().__init__()

    def fetch(self, force=False, **kwargs) -> NDArray[np.int_]:
        """Returns a numpy array of integers representing the median annual household income for each county"""
        if force:
            uncached = self.type_check(kwargs)
            cache_df = DataFrame()
        else:
            uncached, cache_df = self.cache_fetch(kwargs)

        code_string = ','.join(uncached)

        if len(uncached) > 0:
            # fetch data from census
            data = self.census.acs5.get('B19013_001E', {
                                        'for': 'county: *', 'in': f'state: {code_string}'}, year=self.year)

            data_df = DataFrame.from_records(data)

            # sort data by state and county fips
            data_df = data_df.sort_values(by=['state', 'county'])
            data_df.reset_index(inplace=True)

            self.cache_store(data_df, uncached)
            if len(cache_df.index) > 0:
                data_df = concat([data_df, cache_df])

        else:
            data_df = cache_df

        # convert to numpy array and return
        return data_df['B19013_001E'].to_numpy(dtype=np.int_)
