"""Implement a simple geo with a single population and some basic data. Handy for testing."""
import numpy as np

from epymorph.data import registry
from epymorph.data_shape import Shapes
from epymorph.geo.spec import (LABEL, NO_DURATION, AttribDef, CentroidDType,
                               StaticGeoSpec)
from epymorph.geo.static import StaticGeo


@registry.geo('single_pop')
def load() -> StaticGeo:
    """Load the single_pop geo."""
    spec = StaticGeoSpec(
        attributes=[
            LABEL,
            AttribDef('geoid', np.str_, Shapes.N),
            AttribDef('centroid', CentroidDType, Shapes.N),
            AttribDef('population', np.int64, Shapes.N),
            AttribDef('commuters', np.int64, Shapes.NxN),
        ],
        time_period=NO_DURATION
    )
    return StaticGeo(spec, {
        'label': np.array(['AZ'], dtype=np.str_),
        'geoid': np.array(['04'], dtype=np.str_),
        'centroid': np.array([(-111.856111, 34.566667)], dtype=CentroidDType),
        'population': np.array([100_000], dtype=np.int64),
        'commuters': np.array([[0]], dtype=np.int64)
    })
