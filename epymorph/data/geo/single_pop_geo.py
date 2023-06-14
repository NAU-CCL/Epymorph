import numpy as np

from epymorph.geo import Geo


def load() -> Geo:

    return Geo(
        nodes=1,
        labels=["AZ"],
        data={
            'population': np.array([100000], dtype=np.int_)
        }
    )
