import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, Series, concat

from epymorph.adrio.adrio import ADRIO

query_list = ('B01001_003E',  # population 0-19
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


class PopulationByAge(ADRIO):
    """
    ADRIO to fetch  population in all counties in a provided set of states
    broken down into age brackets 0-19, 20-64, and 64-85+
    """
    year = 2015
    attribute = 'population_by_age'

    def __init__(self) -> None:
        super().__init__()

    def calculate_pop(self, start: int, end: int, location: Series) -> int:
        """Adds up a specified group of integer values from a row of a population dataframe (used to calculate different age bracket totals)"""
        population = 0
        for i in range(start, end):
            population += int(location[i])
        return population

    def fetch(self, force=False, **kwargs) -> NDArray[np.int_]:
        """
        Returns a numpy array of 3 element lists containing the population of each age group 
        from youngest to oldest in each county
        """
        if force:
            uncached = self.type_check(kwargs)
            cache_df = DataFrame()
        else:
            uncached, cache_df = self.cache_fetch(kwargs)

        code_string = ','.join(uncached)

        if len(uncached) > 0:
            # get data from census
            data = self.census.acs5.get(query_list, {
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

        # calculate population of each age bracket and enter into a numpy array to return
        output = np.zeros((len(data_df.index), 3), dtype=np.int_)
        for i, row in data_df.iterrows():
            minor_pop = self.calculate_pop(0, 10, row)
            adult_pop = self.calculate_pop(10, 34, row)
            elderly_pop = self.calculate_pop(34, 46, row)
            output[i] = [minor_pop, adult_pop, elderly_pop]

        return output
