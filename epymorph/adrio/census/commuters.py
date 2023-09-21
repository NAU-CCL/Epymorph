import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from epymorph.adrio.census.adrio_census import ADRIO_census, Granularity


class Commuters(ADRIO_census):
    """ADRIO to fetch county or state level commuting flow data from each node to each other node"""
    attribute = 'commuters'

    def fetch(self) -> NDArray[np.int64]:
        """
        Returns an n length numpy array containing the number of commuters from one node to each other node
        """
        data_df = super().fetch_commuters()

        # state level
        if self.granularity == Granularity.STATE.value:
            # get unique state identifier
            unique_states = ('0' + data_df['res_state_code']).unique()
            state_len = np.count_nonzero(unique_states)

            # create dictionary to be used as array indices
            states_enum = enumerate(unique_states)
            states_dict = dict((j, i) for i, j in states_enum)

            # group and aggregate data
            data_df = data_df.groupby(['res_state_code', 'wrk_state_code'])
            data_df = data_df.agg({'workers': 'sum'})

            # create and return array for each state
            output = np.zeros((state_len, state_len), dtype=np.int64)

            # fill array with commuting data
            for i, row in data_df.iterrows():
                if type(i) is tuple:
                    x = states_dict.get('0' + i[0])
                    y = states_dict.get(i[1])

                    output[x][y] = row['workers']

            return output

        # county level
        else:
            # get unique identifier for each county
            geoid_df = DataFrame()
            geoid_df['geoid'] = '0' + data_df['res_state_code'] + \
                data_df['res_county_code']
            unique_counties = geoid_df['geoid'].unique()

            # create empty output array
            county_len = np.count_nonzero(unique_counties)
            output = np.zeros((county_len, county_len), dtype=np.int64)

            # create dictionary to be used as array indices
            counties_enum = enumerate(unique_counties)
            counties_dict = dict((j, i) for i, j in counties_enum)

            data_df.reset_index(drop=True, inplace=True)

            # fill array with commuting data
            for i in range(len(data_df.index)):
                x = counties_dict.get('0' +
                                      data_df.iloc[i]['res_state_code'] + data_df.iloc[i]['res_county_code'])
                y = counties_dict.get(
                    data_df.iloc[i]['wrk_state_code'] + data_df.iloc[i]['wrk_county_code'])

                output[x][y] = data_df.iloc[i]['workers']

            return output
