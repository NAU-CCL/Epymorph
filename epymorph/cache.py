from epymorph.data import geo_library
from epymorph.geo import GEOBuilder


def cache_geo(geo_name, force: bool) -> int:
    """CLI command handler: cache Geo data from ADRIOs without running simulation"""
    builder = geo_library.get(geo_name)
    if builder is not None:
        builder(force)
    print("Data successfully cached")
    return 0  # exit code: success
