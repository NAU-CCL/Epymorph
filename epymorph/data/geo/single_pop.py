import numpy as np

from epymorph.context import SimDType
from epymorph.geo import CentroidDType
from epymorph.geo.geo import Geo, StaticGeo


def load() -> Geo:
    return StaticGeo.from_values({
        'label': np.array(['AZ'], dtype=np.str_),
        'geoid': np.array(['04'], dtype=np.str_),
        'centroid': np.array([(-111.856111, 34.566667)], dtype=CentroidDType),
        'population': np.array([100_000], dtype=SimDType),
        'commuters': np.array([[0]], dtype=SimDType)
    })
