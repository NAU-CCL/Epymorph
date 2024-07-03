"""Implement a simple geo with a single population and some basic data. Handy for testing."""
import numpy as np

from epymorph.data import registry
from epymorph.data_shape import Shapes
from epymorph.data_type import CentroidDType, CentroidType
from epymorph.geo.spec import LABEL, NO_DURATION, StaticGeoSpec
from epymorph.geo.static import StaticGeo
from epymorph.geography.us_census import StateScope
from epymorph.simulation import AttributeDef


@registry.geo('single_pop')
def load() -> StaticGeo:
    """Load the single_pop geo."""
    spec = StaticGeoSpec(
        attributes=[
            LABEL,
            AttributeDef('geoid', type=str, shape=Shapes.N),
            AttributeDef('centroid', type=CentroidType, shape=Shapes.N),
            AttributeDef('population', type=int, shape=Shapes.N),
            AttributeDef('commuters', type=int, shape=Shapes.NxN),
        ],
        scope=StateScope.in_states_by_code(['AZ'], year=2020),
        time_period=NO_DURATION
    )
    return StaticGeo(spec, {
        'label': np.array(['AZ'], dtype=np.str_),
        'geoid': np.array(['04'], dtype=np.str_),
        'centroid': np.array([(-111.856111, 34.566667)], dtype=CentroidDType),
        'population': np.array([100_000], dtype=np.int64),
        'commuters': np.array([[0]], dtype=np.int64)
    })
