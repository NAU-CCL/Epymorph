import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from epymorph.adrio.adrio import ADRIO


class NameState(ADRIO):
    """ADRIO to fetch the names and state names of every county in a provided set of states"""
    year = 2015
    attribute = 'name and state'

    def __init__(self) -> None:
        super().__init__()

    def fetch(self, **kwargs) -> NDArray[np.str_]:
        """Returns a numpy array of 2 element arrays each containing the name and state name of a county as strings"""
        code_string = self.type_check(kwargs)
        code_string = ','.join(code_string)

        # get data from census
        data = self.census.acs5.get(('NAME'),
                                    {'for': 'county: *',
                                        'in': 'state: {}'.format(code_string)},
                                    year=self.year)

        # sort data by state and county fips
        data_df = DataFrame.from_records(data)
        data_df = data_df.sort_values(by=['state', 'county'])
        data_df.reset_index(inplace=True)

        # store county and state names in numpy array to return
        output = NDArray(len(data_df.index), dtype=tuple)
        for i, row in data_df.iterrows():
            curr_names = row['NAME'].split(',')
            curr_names[1] = curr_names[1].strip()
            output[i] = (curr_names[0], curr_names[1])

        return output
