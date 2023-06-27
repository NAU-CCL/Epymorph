import numpy as np
import pandas as pd
from numpy.typing import NDArray

from epymorph.adrio.adrio import ADRIO

query_list = ('B01001_001E',  # total population
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
              'B01001_049E')


class PopByAge(ADRIO):
    """
    ADRIO to fetch total population in all counties in a provided set of states
    as well as population broken down into age brackets 0-19, 20-64, and 64-85+
    """
    year = 2015
    attribute = 'population'

    def __init__(self) -> None:
        super().__init__()

    # adds up a specified group of integer values from a row of a population dataframe (used to calculate different age bracket totals)
    def calculate_pop(self, start: int, end: int, location: pd.Series) -> int:
        population = 0
        for i in range(start, end):
            population += int(location[i])
        return population

    def fetch(self, **kwargs) -> NDArray[np.int_]:
        """
        Returns a numpy array of 4 element lists containing each county's total population
        followed by the population of each age group from youngest to oldest
        """
        code_string = self.type_check(kwargs)
        code_string = ','.join(code_string)

        # get data from census
        data = self.census.acs5.get(query_list, {
                                    'for': 'county: *', 'in': 'state: {}'.format(code_string)}, year=self.year)

        data_df = pd.DataFrame.from_records(data)

        # sort data by state and county fips
        data_df = data_df.sort_values(by=['state', 'county'])
        data_df.reset_index(inplace=True)

        # calculate population of each age bracket and enter into a numpy array to return
        output = np.zeros((len(data_df.index), 4), dtype=np.int_)
        for i, rows in data_df.iterrows():
            total_pop = rows['B01001_001E']
            minor_pop = self.calculate_pop(1, 11, rows)
            adult_pop = self.calculate_pop(11, 35, rows)
            elderly_pop = self.calculate_pop(35, 47, rows)
            output[i] = [total_pop, minor_pop, adult_pop, elderly_pop]

        return output
