import os
from pathlib import Path
from platform import system

from epymorph.data import geo_library_dynamic

# TODO: add case for Mac
if system() == 'Linux':
    CACHE_PATH = Path(os.path.expanduser('~')) / '.epymorph' / 'cache'
else:
    CACHE_PATH = Path(os.path.expanduser('~')) / 'AppData' / \
        'local' / 'epymorph' / 'cache'


# Exit codes:
# - 0 success
# - 1 invalid input
def cache_geo(geo_name: str | None, force: bool, remove: bool, list: bool, clear: bool) -> int:
    """CLI command handler: cache Geo data from ADRIOs without running simulation"""
    # make cache directory if needed
    if not os.path.exists(Path(f'{CACHE_PATH}')):
        os.makedirs(Path(f'{CACHE_PATH}'))

    # check for invalid arguments
    unique_args = 0
    if geo_name is not None:
        unique_args += 1
        if force and remove:
            unique_args += 1
    if list:
        unique_args += 1
    if clear:
        unique_args += 1

    if unique_args > 1:
        print('Error: several incompatible flags passed. Please choose only one operation to execute')
        return 1  # exit code: invalid args
    elif unique_args == 0:
        print('Error: please specify a Geo to cache or cache operation to perform')
        return 1  # exit code: invalid args

    # assume operation is being done on single geo
    if geo_name is not None:
        # cache specified geo
        if not remove:
            choice = 'n'
            if os.path.exists(Path(f'{CACHE_PATH}/{geo_name}_geo.npz')) and not force:
                choice = input(f'{geo_name} is already cached, overwrite? [y/n]')
            if force or choice == 'y':
                geo_load = geo_library_dynamic.get(geo_name)
                if geo_load is not None:
                    geo = geo_load()
                    geo.save(Path(f'{CACHE_PATH}/{geo_name}_geo.npz'))
                    print('Geo sucessfully cached.')
        # remove specified geo
        else:
            os.remove(Path(f'{CACHE_PATH}/{geo_name}_geo.npz'))
            print(f'{geo_name} removed from cache.')

    # list geos in cache
    elif list:
        for file in os.listdir(Path(f'{CACHE_PATH}')):
            print(file.removesuffix('_geo.npz'))

    # clear geo cache
    elif clear:
        for file in os.listdir(Path(f'{CACHE_PATH}')):
            os.remove(Path(f'{CACHE_PATH}/{file}'))
        print('cleared Geo cache.')

    return 0  # exit code: success
