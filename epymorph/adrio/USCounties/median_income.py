import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from epymorph.adrio.adrio import ADRIO


class MedianIncome(ADRIO):
    """ADRIO to fetch median household income for each county in a provided set of states"""
    year = 2015
    attribute = 'median income'

    def __init__(self) -> None:
        super().__init__()

    def fetch(self, **kwargs) -> NDArray[np.int_]:
        """Returns a numpy array of integers representing the median annual household income for each county"""
        code_string = self.type_check(kwargs)
        code_string = ','.join(code_string)

        # fetch data from census
        data = self.census.acs5.get('B19013_001E', {
                                    'for': 'county: *', 'in': 'state: {}'.format(code_string)}, year=self.year)

        # sort data by state and county fips
        data_df = DataFrame.from_records(data)
        data_df = data_df.sort_values(by=['state', 'county'])

        # convert to numpy array and return
        return data_df['B19013_001E'].to_numpy(dtype=np.int_)
