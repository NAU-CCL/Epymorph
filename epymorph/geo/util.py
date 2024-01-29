"""Utility functions for interacting with geos of various types."""
from epymorph.geo.dynamic import DynamicGeo, FetchStart
from epymorph.geo.spec import StaticGeoSpec
from epymorph.geo.static import StaticGeo


def convert_to_static_geo(geo: DynamicGeo) -> StaticGeo:
    """
    Convert a DynamicGeo to a StaticGeo, proactively fetching all of its values.
    """
    spec = StaticGeoSpec(
        attributes=geo.spec.attributes,
        time_period=geo.spec.time_period,
    )
    geo.fetch_start.publish(FetchStart(len(geo._adrios)))
    geo.fetch_all()
    values = {
        attr.name: geo[attr.name]
        for attr in geo.spec.attributes
    }
    geo.fetch_end.publish(None)
    return StaticGeo(spec, values)
