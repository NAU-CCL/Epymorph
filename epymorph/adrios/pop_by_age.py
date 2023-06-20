import numpy as np
from census import Census
from numpy.typing import NDArray

from epymorph.adrio import ADRIO


class PopByAge(ADRIO):
    """
    ADRIO to fetch total population in all counties in a provided set of states
    as well as population broken down into age brackets 0-19, 20-64, and 64-85+
    Returns an array of 4 element lists containing each county's total population
    followed by the population of each age group from youngest to oldest
    """
    census: Census
    year = 2015
    attribute = 'population'

    def __init__(self, key: str) -> None:
        self.census = Census(key)

    # adds up values in an integer list in the range provided (used to calculate different age bracket totals)
    def calculate_pop(self, start: int, end: int, location: list[int]) -> int:
        population = 0
        for i in range(start, end + 1):
            population += location[i]
        return population

    def fetch(self, **kwargs) -> NDArray:
        code_string = self.format_geo_codes(kwargs)

        # get data from census (census package not working with subject tables? 2015 data in percentages?)
        # This doesn't work
        """data = self.census.acs5.get(('NAME',
                                     'S0101_C01_001E',  # total population
                                     'S0101_C01_002E',  # population 0-19
                                     'S0101_C01_003E',
                                     'S0101_C01_004E',
                                     'S0101_C01_005E',
                                     'S0101_C01_006E',  # population 20-64
                                     'S0101_C01_007E',
                                     'S0101_C01_008E',
                                     'S0101_C01_009E',
                                     'S0101_C01_010E',
                                     'S0101_C01_011E',
                                     'S0101_C01_012E',
                                     'S0101_C01_013E',
                                     'S0101_C01_014E',
                                     'S0101_C01_015E',  # population 65-85+
                                     'S0101_C01_016E',
                                     'S0101_C01_017E',
                                     'S0101_C01_018E',
                                     'S0101_C01_019E'),
                                    {'for': 'county: *',
                                     'in': 'state: {}'.format(code_string)},
                                    year=self.year)"""

        # roundabout solution with more fetching and calculations (works)
        data = self.census.acs5.get(('NAME',
                                     'B01001_001E',  # total population
                                     'B01001_003E',  # population 0-19
                                     'B01001_004E',
                                     'B01001_005E',
                                     'B01001_006E',
                                     'B01001_007E',
                                     'B01001_027E',  # women
                                     'B01001_028E',
                                     'B01001_029E',
                                     'B01001_030E',
                                     'B01001_031E',
                                     'B01001_008E',  # population 20-64
                                     'B01001_009E',
                                     'B01001_010E',
                                     'B01001_011E',
                                     'B01001_012E',
                                     'B01001_013E',
                                     'B01001_014E',
                                     'B01001_015E',
                                     'B01001_016E',
                                     'B01001_017E',
                                     'B01001_018E',
                                     'B01001_019E',
                                     'B01001_032E',  # women
                                     'B01001_033E',
                                     'B01001_034E',
                                     'B01001_035E',
                                     'B01001_036E',
                                     'B01001_037E',
                                     'B01001_038E',
                                     'B01001_039E',
                                     'B01001_040E',
                                     'B01001_041E',
                                     'B01001_042E',
                                     'B01001_043E',
                                     'B01001_020E',  # population 65-85+
                                     'B01001_021E',
                                     'B01001_022E',
                                     'B01001_023E',
                                     'B01001_024E',
                                     'B01001_025E',
                                     'B01001_044E',  # women
                                     'B01001_045E',
                                     'B01001_046E',
                                     'B01001_047E',
                                     'B01001_048E',
                                     'B01001_049E',),
                                    {'for': 'county: *',
                                        'in': 'state: {}'.format(code_string)},
                                    year=self.year)

        # sort data by state and county fips
        datalist = self.sort_counties(data)

        # calculate population of each age bracket and enter into a numpy array to return
        output = np.zeros((len(data), 4), dtype=np.int_)
        for i in range(len(datalist)):
            total_pop = datalist[i][1]
            minor_pop = self.calculate_pop(2, 11, datalist[i])
            adult_pop = self.calculate_pop(12, 35, datalist[i])
            elderly_pop = self.calculate_pop(36, 47, datalist[i])
            output[i] = [total_pop, minor_pop, adult_pop, elderly_pop]

        return output
