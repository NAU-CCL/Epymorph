import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, concat

from epymorph.adrio.adrio import ADRIO


class GEOID(ADRIO):
    """ADRIO to fetch the geoid (state fips + county fips) of all counties in a provided set of states"""
    year = 2015
    attribute = 'geoid'

    def __init__(self) -> None:
        super().__init__()

    def fetch(self, force=False, **kwargs) -> NDArray[np.str_]:
        """Returns a numpy array of containing the geoid of every county as strings"""
        if force:
            uncached = self.type_check(kwargs)
            cache_df = DataFrame()
        else:
            uncached, cache_df = self.cache_fetch(kwargs)

        # check for uncached data
        if len(uncached) > 0:
            code_string = ','.join(uncached)

            # get data from census
            data = self.census.acs5.get(('NAME'),
                                        {'for': 'county: *',
                                         'in': f'state: {code_string}'},
                                        year=self.year)

            data_df = DataFrame.from_records(data)
            # strange interaction here - name field is fetched only because a field is required
            data_df = data_df.drop(columns='NAME')

            # sort data by state and county fips
            data_df = data_df.sort_values(by=['state', 'county'])
            data_df.reset_index(inplace=True)

            # cache data
            self.cache_store(data_df, uncached)

            # join with data fetched from cache if there is any
            if len(cache_df.index) > 0:
                data_df = concat([data_df, cache_df])

        else:
            data_df = cache_df

        # concatenate state and county fips to yield geoid
        output = list()
        for i, row in data_df.iterrows():
            output.append(row['state'] + row['county'])

        return np.array(output, dtype=np.str_)
