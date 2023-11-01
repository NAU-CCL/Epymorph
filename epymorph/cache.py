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
        choice = input(f'{geo_name} is already cached, overwrite? [y/n]')
    if force or choice == 'y':
        result = cache.fetch(geo_name)
        if result:
            print('geo sucessfully cached.')
        else:
            print(f'spec file for {geo_name} not found.')
            return 1  # exit code: geo not found
    return 0  # exit code: success


def remove(geo_name: str) -> int:
    """CLI command handler: remove geo from cache"""
    result = cache.remove(geo_name)
    if result:
        print(f'{geo_name} removed from cache.')
        return 0  # exit code: success
    else:
        print(f'{geo_name} not found in cache, check your spelling or use the list subcommand to view all currently cached geos')
        return 1  # exit code: not found


def list() -> int:
    num_geos = len(os.listdir(cache.CACHE_PATH))
    if num_geos > 0:
        print(
            f'epymorph geo cache contains {num_geos} geos totaling {cache.get_total_size()} ({cache.CACHE_PATH})')
        cache.list()
    else:
        print(f'epymorph geo cache is empty ({cache.CACHE_PATH})')
    return 0  # exit code: success


def clear() -> int:
    """CLI command handler: clear geo cache"""
    if len(os.listdir(cache.CACHE_PATH)) > 0:
        print(
            f'The following geos will be removed from the cache ({cache.CACHE_PATH}) and free {cache.get_total_size()} of space:')
        cache.list()
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
