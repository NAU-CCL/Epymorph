import numpy as np
from numpy.typing import NDArray

from epymorph.adrio.adrio_census import ADRIO_census


class GEOID(ADRIO_census):
    """ADRIO to fetch the geoid (state fips + county fips) of all counties in a provided set of states"""
    attribute = 'geoid'

    def fetch(self) -> NDArray[np.str_]:
        """Returns a numpy array of containing the geoid of every county as strings"""
        data_df = super().fetch(['NAME'])

        # strange interaction here - name field is fetched only because a field is required
        data_df = data_df.drop(columns='NAME')

        # concatenate state and county fips to yield geoid
        output = list()
        for i, row in data_df.iterrows():
            output.append(row['state'] + row['county'])

        return np.array(output, dtype=np.str_)
