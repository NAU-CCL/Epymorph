from epymorph.geo import GEOBuilder


def cache_geo(file_path, force: bool) -> int:
    """CLI command handler: cache Geo data from ADRIOs without running simulation"""
    try:
        builder = GEOBuilder(file_path)
        builder.build(force)
    except Exception as e:
        print(f"[✗] Invalid specification file: {file_path}")
        print(e)
        return 3  # exit code: invalid spec
    print("Data successfully cached")
    return 0  # exit code: success
