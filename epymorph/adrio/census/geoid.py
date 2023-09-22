import numpy as np
from numpy.typing import NDArray

from epymorph.adrio.census.adrio_census import ADRIO_census, Granularity


class GEOID(ADRIO_census):
    """ADRIO to fetch the geoid of all geographies provided"""
    attribute = 'geoid'

    def fetch(self) -> NDArray[np.str_]:
        """Returns a numpy array of containing the geoid of every node as strings"""
        data_df = super().fetch(['NAME'])

        # strange interaction here - name field is fetched only because a field is required
        data_df = data_df.drop(columns='NAME')

        # concatenate individual fips codes to yield geoid
        output = list()
        for i in range(len(data_df.index)):
            # state geoid is the same as fips code - no action required
            if self.granularity == Granularity.STATE.value:
                output.append(str(data_df.loc[i, 'state']))
            elif self.granularity == Granularity.COUNTY.value:
                output.append(str(data_df.loc[i, 'state']) +
                              str(data_df.loc[i, 'county']))
            elif self.granularity == Granularity.TRACT.value:
                output.append(str(
                    data_df.loc[i, 'state']) + str(data_df.loc[i, 'county']) + str(data_df.loc[i, 'tract']))
            else:
                output.append(str(data_df.loc[i, 'state']) + str(data_df.loc[i, 'county']) + str(
                    data_df.loc[i, 'tract']) + str(data_df.loc[i, 'block group']))

        return np.array(output, dtype=np.str_)
