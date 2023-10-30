import os

from epymorph.geo.cache import (CACHE_PATH, cache_clear, cache_fetch,
                                cache_list, cache_remove)
from epymorph.geo.static import StaticGeoFileOps

# Exit codes:
# - 0 success
# - 1 geo not found
# - 2 no cache directory


def cache_fetch_CLI(geo_name: str, force: bool) -> int:
    """CLI command handler: cache dynamic Geo data."""
    # cache specified geo
    filepath = CACHE_PATH / StaticGeoFileOps.get_tar_filename(geo_name)
    choice = 'y'
    if os.path.exists(filepath) and not force:
        choice = input(f'{geo_name} is already cached, overwrite? [y/n]')
    if force or choice == 'y':
        cache_fetch(geo_name)
        print('Geo sucessfully cached.')

    return 0  # exit code: success


def cache_remove_CLI(geo_name: str) -> int:
    """CLI command handler: remove Geo from cache"""
    if not os.path.exists(CACHE_PATH):
        print('cache directory not found')
        return 2  # exit code: no cache

    filepath = CACHE_PATH / StaticGeoFileOps.get_tar_filename(geo_name)

    if not os.path.exists(filepath):
        print(f'{geo_name} not found in cache, check your spelling or use the list subcommand to view all currently cached Geos')
        return 1  # exit code: not found
    else:
        cache_remove(geo_name)
        print(f'{geo_name} removed from cache.')
        return 0  # exit code: success


def cache_list_CLI() -> int:
    """CLI command handler: list Geos in cache"""
    if not os.path.exists(CACHE_PATH):
        print('cache directory not found')
        return 2  # exit code: no cache

    else:
        cache_list()
        return 0  # exit code: success


def cache_clear_CLI() -> int:
    """CLI command handler: clear Geo cache"""
    if not os.path.exists(CACHE_PATH):
        print('cache directory not found')
        return 2  # exit code: no cache

    else:
        cache_clear()
        print('cleared Geo cache.')
        return 0  # exit code: success
