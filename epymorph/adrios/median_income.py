import numpy as np
from census import Census
from numpy.typing import NDArray

from epymorph.adrio import ADRIO


class MedianIncome(ADRIO):
    """
    ADRIO to fetch median household income for each county in a provided set of states
    Returns an array of integers representing the median annual household income for each county
    """
    census: Census
    year = 2015
    attribute = 'median income'

    def __init__(self, key: str) -> None:
        self.census = Census(key)

    def fetch(self, **kwargs) -> NDArray[np.int_]:
        code_string = self.format_geo_codes(kwargs)

        # fetch data from census
        data = self.census.acs5.get(("NAME", "B19013_001E"), {
                                    'for': 'county: *', 'in': 'state: {}'.format(code_string)}, year=self.year)

        # sort data by state and county fips
        datalist = self.sort_counties(data)

        # map incomes to county name in a numpy array and return
        output = np.zeros((len(datalist),), dtype=np.int_)
        for i in range(len(datalist)):
            output[i] = int(datalist[i][1])

        return output
