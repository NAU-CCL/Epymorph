import os
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from us.states import lookup

from epymorph.error import GeoValidationException
from epymorph.geo.adrio.adrio import ADRIO, ADRIOMaker
from epymorph.geo.adrio.census.adrio_census import CensusGeography
from epymorph.geo.spec import AttribDef, Geography


class ADRIOMakerFile(ADRIOMaker):

    def make_adrio(self, attrib: AttribDef, geography: Geography, file_path: str, key: str) -> ADRIO:
        def fetch() -> NDArray:
            # check if file exists
            if os.path.exists(file_path):
                path = Path(file_path)
                states = []  # sort key
                # lookup state by abbreviation and sort by fips
                if key == 'state_abbr' and isinstance(geography, CensusGeography):
                    state_fips = geography.filter.get('state')
                    if state_fips is not None:
                        for fip in state_fips:
                            state = lookup(fip)
                            if state is not None:
                                states.append(state.abbr)

                # read value from csv
                if path.suffix == '.csv':
                    dataframe = pd.read_csv(path, header=0)

                    # drop unneeded columns and sort
                    dataframe = dataframe.loc[dataframe['state_name'].isin(states)]
                    dataframe.set_index('state_name')
                    states = pd.DataFrame({'state_name': states})
                    dataframe = pd.merge(states, dataframe, how='left')
                    if dataframe['population'].isnull().any():
                        msg = f"Data for required geographies missing in {attrib.name} attribute file or could not be found."
                        raise GeoValidationException(msg)

                    return dataframe['population'].to_numpy(dtype=np.int64)

                # read value from npz
                elif path.suffix == '.npz':
                    return np.load(path)[attrib.name]

                # raise exception for any other file type
                else:
                    msg = "Invalid file type. Supported file types are .csv and .npz"
                    raise Exception(msg)
            else:
                msg = f"File {file_path} not found"
                raise Exception(msg)

        return ADRIO(attrib.name, fetch)
