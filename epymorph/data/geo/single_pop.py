"""Implement a simple geo with a single population and some basic data. Handy for testing."""
import numpy as np

from epymorph.context import SimDType
from epymorph.geo.common import CentroidDType
from epymorph.geo.geo import Geo
from epymorph.geo.static import StaticGeo


def load() -> Geo:
    """Load the single_pop geo."""
    return StaticGeo.from_values({
        'label': np.array(['AZ'], dtype=np.str_),
        'geoid': np.array(['04'], dtype=np.str_),
        'centroid': np.array([(-111.856111, 34.566667)], dtype=CentroidDType),
        'population': np.array([100_000], dtype=SimDType),
        'commuters': np.array([[0]], dtype=SimDType)
    })
