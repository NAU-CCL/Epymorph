"""Utility functions for interacting with geos of various types."""
from epymorph.geo.dynamic import DynamicGeo
from epymorph.geo.spec import StaticGeoSpec
from epymorph.geo.static import StaticGeo


def convert_to_static_geo(geo: DynamicGeo) -> StaticGeo:
    """
    Convert a DynamicGeo to a StaticGeo, proactively fetching all of its values.
    """
    spec = StaticGeoSpec(
        attributes=geo.spec.attributes,
        scope=geo.spec.scope,
        time_period=geo.spec.time_period,
    )
    geo.fetch_all()
    values = {
        attr.name: geo[attr.name]
        for attr in geo.spec.attributes
    }
    return StaticGeo(spec, values)
