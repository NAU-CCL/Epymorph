import os
from pathlib import Path
from platform import system

from epymorph.data import geo_library_dynamic
from epymorph.geo.static import StaticGeoFileOps
from epymorph.geo.util import convert_to_static_geo

# TODO: add case for Mac
if system() == 'Linux':
    CACHE_PATH = Path(os.path.expanduser('~')) / '.epymorph' / 'cache'
else:
    CACHE_PATH = Path(os.path.expanduser('~')) / 'AppData' / \
        'local' / 'epymorph' / 'cache'


# Exit codes:
# - 0 success
# - 1 geo not found
# - 2 no cache directory


def cache_fetch(geo_name: str, force: bool) -> int:
    """CLI command handler: cache Geo data from ADRIOs without running simulation"""
    # make cache directory if needed
    if not os.path.exists(CACHE_PATH):
        os.makedirs(CACHE_PATH)

    # cache specified geo
    filepath = CACHE_PATH / StaticGeoFileOps.get_tar_filename(geo_name)
    choice = 'y'
    if os.path.exists(filepath) and not force:
        choice = input(f'{geo_name} is already cached, overwrite? [y/n]')
    if force or choice == 'y':
        geo_load = geo_library_dynamic.get(geo_name)
        if geo_load is not None:
            geo = geo_load()
            static_geo = convert_to_static_geo(geo)
            static_geo.save(filepath)
            print('Geo sucessfully cached.')

    return 0  # exit code: success


def cache_remove(geo_name: str) -> int:
    """CLI command handler: remove Geo from cache"""
    if not os.path.exists(CACHE_PATH):
        print('cache directory not found')
        return 2  # exit code: no cache

    filepath = CACHE_PATH / StaticGeoFileOps.get_tar_filename(geo_name)

    if not os.path.exists(filepath):
        print(f'{geo_name} not found in cache, check your spelling or use the list subcommand to view all currently cached Geos')
        return 1  # exit code: not found
    else:
        os.remove(filepath)
        print(f'{geo_name} removed from cache.')
        return 0  # exit code: success


def cache_list() -> int:
    """CLI command handler: list Geos in cache"""
    if not os.path.exists(CACHE_PATH):
        print('cache directory not found')
        return 2  # exit code: no cache

    else:
        files = os.listdir(CACHE_PATH)
        if len(files) == 0:
            print('cache is empty')
        else:
            for file in files:
                print(file.removesuffix('.geo.tar'))
        return 0  # exit code: success


def cache_clear() -> int:
    """CLI command handler: clear Geo cache"""
    if not os.path.exists(CACHE_PATH):
        print('cache directory not found')
        return 2  # exit code: no cache

    else:
        for file in os.listdir(Path(f'{CACHE_PATH}')):
            os.remove(CACHE_PATH / file)
        print('cleared Geo cache.')
        return 0  # exit code: success
