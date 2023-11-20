import os

from epymorph.geo import cache
from epymorph.geo.static import StaticGeoFileOps as F

# Exit codes:
# - 0 success
# - 1 geo not found
# - 2 empty cache


def fetch(geo_name: str, force: bool) -> int:
    """CLI command handler: cache dynamic geo data."""
    # cache specified geo
    filepath = cache.CACHE_PATH / F.to_archive_filename(geo_name)
    choice = 'y'
    if os.path.exists(filepath) and not force:
        choice = input(f'{geo_name} is already cached, overwrite? [y/n] ')
    if force or choice == 'y':
        try:
            cache.fetch(geo_name)
            print('geo sucessfully cached.')
        except cache.GeoCacheException as e:
            print(e)
            return 1  # exit code: geo not found
    return 0  # exit code: success


def remove(geo_name: str) -> int:
    """CLI command handler: remove geo from cache"""
    try:
        cache.remove(geo_name)
        print(f'{geo_name} removed from cache.')
        return 0  # exit code: success
    except cache.GeoCacheException as e:
        print(e)
        return 1  # exit code: not found


def print_geos() -> int:
    """CLI command handler: print geo cache information"""
    geos = cache.list_geos()
    num_geos = len(geos)
    if num_geos > 0:
        print(
            f'epymorph geo cache contains {num_geos} geos totaling {cache.get_total_size()} ({cache.CACHE_PATH})')
        for (name, file_size) in geos:
            print(f"* {name} ({cache.format_size(file_size)})")
    else:
        print(f'epymorph geo cache is empty ({cache.CACHE_PATH})')
    return 0  # exit code: success


def clear() -> int:
    """CLI command handler: clear geo cache"""
    if len(os.listdir(cache.CACHE_PATH)) > 0:
        print(
            f'The following geos will be removed from the cache ({cache.CACHE_PATH}) and free {cache.get_total_size()} of space:')
        for (name, file_size) in cache.list_geos():
            print(f"* {name} ({cache.format_size(file_size)})")
        choice = input('proceed? [y/n] ')
        if choice == 'y':
            cache.clear()
            print('cleared geo cache.')
        else:
            print('cache clear aborted.')

        return 0  # exit code: success
    else:
        print(f'cache ({cache.CACHE_PATH}) is empty, nothing to clear.')
        return 2  # exit code: empty cache
