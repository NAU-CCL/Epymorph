import numpy as np
from census import Census
from numpy.typing import NDArray

from epymorph.adrio import ADRIO


class dissimilarityIndex(ADRIO):
    """
    ADRIO to fetch the dissimilarity index of racial segregation for all counties in a provided set of states
    Returns an array of floats representing the disimilarity index for each county
    """
    census: Census
    year = 2015
    attribute = 'dissimilarity index'

    def __init__(self, key: str):
        self.census = Census(key)

    def fetch(self, **kwargs) -> NDArray[np.float_]:
        code_string = self.format_geo_codes(kwargs)

        # fetch tract level data from census
        tract_data = self.census.acs5.get(('B03002_003E',  # white population
                                           'B03002_013E',
                                           'B03002_004E',  # minority population
                                           'B03002_014E'),
                                          {'for': 'tract: *', 'in': 'county: *',
                                           'in': 'state: {}'.format(code_string)},
                                          year=self.year)

        # fetch county level data from census
        county_data = self.census.acs5.get(('B03002_003E',  # white population
                                            'B03002_013E',
                                            'B03002_004E',  # minority population
                                            'B03002_014E'),
                                           {'for': 'county: *',
                                               'in': 'state: {}'.format(code_string)},
                                           year=self.year)

        # sort data by state, county
        tract_datalist = self.sort_counties(tract_data)

        county_datalist = self.sort_counties(county_data)

        output = np.zeros((len(county_data), 1), dtype=np.float_)
        j = 0
        # loop for counties
        for i in range(len(county_data)):
            # assign county fip to variable
            tract_fip = tract_datalist[i][6]
            # loop for tracts (while fip == variable)
            sum = 0.0
            while (tract_datalist[j][6] == tract_fip):
                # run calculation sum += ( |minority(tract) / minority(county) - majority(tract) / majority(county)| )
                sum = sum + abs(tract_datalist[j][0] / county_datalist[i]
                                [0] - tract_datalist[j][1] / county_datalist[i][1])
                j = j + 1
            # sum = sum * .5
            sum = sum * .5
            # assign current output element to sum
            output[i] = sum

        return output
