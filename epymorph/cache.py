from epymorph.data import geo_library
from epymorph.geo import GEOBuilder


def cache_geo(geo_name, force: bool) -> int:
    """CLI command handler: cache Geo data from ADRIOs without running simulation"""
    builder = geo_library.get(geo_name)
    if builder is not None:
        builder(force)
    else:
        print("The specified Geo does not exist or could not be retrieved")
        return 3  # exit code: invalid geo
    print("Data successfully cached")
    return 0  # exit code: success
