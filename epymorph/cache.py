from epymorph.data import geo_library_dynamic


def cache_geo(geo_name, force: bool) -> int:
    """CLI command handler: cache Geo data from ADRIOs without running simulation"""
    builder = geo_library_dynamic.get(geo_name)
    if builder is not None:
        builder(force)
    else:
        print("The specified Geo does not exist, could not be retrieved, or uses data that is not cachable")
        return 3  # exit code: invalid geo
    print("Data successfully cached")
    return 0  # exit code: success
