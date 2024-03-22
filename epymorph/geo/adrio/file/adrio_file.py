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

    def make_adrio(self, attrib: AttribDef, geography: Geography, file_path: str, label_key: str | int, value_key: str | int, join: str, header: int | None = None) -> ADRIO:
        def fetch() -> NDArray:
            # check if file exists
            if os.path.exists(file_path):
                path = Path(file_path)
                states = list()  # sort key
                # lookup state by abbreviation and sort by fips
                if join == 'state_abbr' and isinstance(geography, CensusGeography):
                    state_fips = geography.filter.get('state')
                    if state_fips is not None:
                        for fip in state_fips:
                            state = lookup(fip)
                            if state is not None:
                                states.append(state.abbr)

                # read value from csv
                if path.suffix == '.csv':
                    dataframe = pd.read_csv(path, header=header)

                    # column name passed
                    if isinstance(label_key, str):
                        dataframe = dataframe.loc[dataframe[label_key].isin(states)]
                        states = pd.DataFrame({label_key: states})
                        dataframe = pd.merge(states, dataframe, how='left')
                    # column index passed (currently only works for no header)
                    elif isinstance(label_key, int):
                        dataframe = dataframe.loc[dataframe.iloc[:, label_key].isin(
                            states)]
                        states = pd.DataFrame(states)
                        dataframe = pd.merge(
                            states, dataframe, how='left', on=label_key)
                    # check for null values (missing data in file)
                    if dataframe[value_key].isnull().any():
                        msg = f"Data for required geographies missing from {attrib.name} attribute file or could not be found."
                        raise GeoValidationException(msg)

                    return dataframe[value_key].to_numpy(dtype=np.int64)

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
