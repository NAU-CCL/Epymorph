import numpy as np
from numpy.typing import NDArray
from pandas import Series

from epymorph.adrio.census.adrio_census import ADRIO_census

query_list = ['B01001_003E',  # population 0-19
              'B01001_004E',
              'B01001_005E',
              'B01001_006E',
              'B01001_007E',
              'B01001_027E',  # women
              'B01001_028E',
              'B01001_029E',
              'B01001_030E',
              'B01001_031E',
              'B01001_008E',  # population 20-34
              'B01001_009E',
              'B01001_010E',
              'B01001_011E',
              'B01001_012E',
              'B01001_032E',  # women
              'B01001_033E',
              'B01001_034E',
              'B01001_035E',
              'B01001_036E',
              'B01001_013E',  # population 35-54
              'B01001_014E',
              'B01001_015E',
              'B01001_016E',
              'B01001_037E',  # women
              'B01001_038E',
              'B01001_039E',
              'B01001_040E',
              'B01001_017E',  # population 55-64
              'B01001_018E',
              'B01001_019E',
              'B01001_041E',  # women
              'B01001_042E',
              'B01001_043E',
              'B01001_020E',  # population 65-74
              'B01001_021E',
              'B01001_022E',
              'B01001_044E',  # women
              'B01001_045E',
              'B01001_046E',
              'B01001_023E',  # population 75+
              'B01001_024E',
              'B01001_025E',
              'B01001_047E',  # women
              'B01001_048E',
              'B01001_049E']


class PopulationByAgex6(ADRIO_census):
    """
    ADRIO to fetch population in a provided set of geographies
    broken down into age brackets 0-19, 20-34, 35-54, 55-64, 65-74, and 75+
    """
    attribute = 'population_by_age_x6'

    def calculate_pop(self, start: int, end: int, location: Series) -> int:
        """
        Adds up a specified group of integer values from a row of a population dataframe
        (Used to calculate different age bracket totals)
        """
        population = 0
        for i in range(start, end):
            population += int(location[i])
        return population

    def fetch(self) -> NDArray[np.int_]:
        """
        Returns a numpy array of 6 element lists containing the population of each age group 
        from youngest to oldest in each node
        """
        # get data from census
        data_df = super().fetch(query_list)

        # calculate population of each age bracket and enter into a numpy array to return
        output = np.zeros((len(data_df.index), 6), dtype=np.int_)
        for i, row in data_df.iterrows():
            pop1 = self.calculate_pop(0, 10, row)
            pop2 = self.calculate_pop(10, 20, row)
            pop3 = self.calculate_pop(20, 28, row)
            pop4 = self.calculate_pop(28, 34, row)
            pop5 = self.calculate_pop(34, 40, row)
            pop6 = self.calculate_pop(40, 46, row)
            output[i] = [pop1, pop2, pop3, pop4, pop5, pop6]

        return output
