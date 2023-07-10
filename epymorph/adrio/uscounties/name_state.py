import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, concat

from epymorph.adrio.adrio import ADRIO


class NameState(ADRIO):
    """ADRIO to fetch the names and state names of every county in a provided set of states"""
    year = 2015
    attribute = 'name and state'

    def __init__(self) -> None:
        super().__init__()

    def fetch(self, force=False, **kwargs) -> NDArray[np.str_]:
        """Returns a numpy array of 2 element arrays each containing the name and state name of a county as strings"""
        if force:
            code_list = self.type_check(kwargs)
            cache_df = DataFrame()
        else:
            cache_data = self.cache_fetch(kwargs, self.attribute)
            code_list = cache_data[0]
            cache_df = cache_data[1]

        # check for uncached data
        if len(code_list) > 0:
            code_string = ','.join(code_list)

            # get data from census
            data = self.census.acs5.get(('NAME'),
                                        {'for': 'county: *',
                                         'in': f'state: {code_string}'},
                                        year=self.year)

            data_df = DataFrame.from_records(data)

            # cache data
            self.cache_store(data_df, code_list, self.attribute)

            # join with data fetched from cache if there is any
            if len(cache_df.index) > 0:
                data_df = concat([data_df, cache_df])

        else:
            data_df = cache_df

        # sort data by state and county fips
        data_df = data_df.sort_values(by=['state', 'county'])
        data_df.reset_index(inplace=True)

        # store county and state names in numpy array to return
        output = NDArray(len(data_df.index), dtype=tuple)
        for i, row in data_df.iterrows():
            curr_names = row['NAME'].split(',')
            curr_names[1] = curr_names[1].strip()
            output[i] = (curr_names[0], curr_names[1])

        return output
