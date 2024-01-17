import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from epymorph.geo.adrio.adrio import ADRIO, ADRIOMaker
from epymorph.geo.spec import AttribDef, Geography, TimePeriod


class ADRIOMakerFile(ADRIOMaker):

    def make_adrio(self, attrib: AttribDef, geography: Geography, time_period: TimePeriod, file_path: Path, key: int) -> ADRIO:
        def fetch() -> NDArray:
            # check if file exists
            if os.path.exists(file_path):
                # read value from csv
                if file_path.suffix == '.csv':
                    return np.loadtxt(file_path, usecols=key)
                # read value from npz
                elif file_path.suffix == '.npz':
                    return np.load(file_path)[attrib.name]
                # raise exception for any other file type
                else:
                    msg = "Invalid file type. Supported file types are .csv and .npz"
                    raise Exception(msg)
            else:
                msg = f"File {file_path} not found"
                raise Exception(msg)

        return ADRIO(attrib.name, fetch)
