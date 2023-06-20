import numpy as np
from census import Census
from numpy.typing import NDArray

from epymorph.adrio import ADRIO


class nameState(ADRIO):
    """
    ADRIO to fetch the names and state names of every county in a provided set of states
    Returns an array of 2 element arrays each containing the name and state name of a county as strings
    """
    census: Census
    year = 2015
    attribute = 'name and state'

    def __init__(self, key: str) -> None:
        self.census = Census(key)

    def fetch(self, **kwargs) -> NDArray:
        code_string = self.format_geo_codes(kwargs)

        # get data from census
        data = self.census.acs5.get(('NAME'),
                                    {'for': 'county: *',
                                        'in': 'state: {}'.format(code_string)},
                                    year=self.year)

        # sort data by state and county fips
        datalist = self.sort_counties(data)

        # store county and state names in numpy array to return
        # output = NDArray((len(datalist), 2), dtype=np.str_) - not working?
        output = [[0] * 2] * len(datalist)
        for i in range(len(datalist)):
            curr_names = datalist[i][0].split(',')
            output[i] = [curr_names[0], curr_names[1]]

        output = np.array(output)
        return output
