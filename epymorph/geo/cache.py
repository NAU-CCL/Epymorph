import os
from pathlib import Path
from platform import system

from epymorph.data import geo_library_dynamic
from epymorph.geo.geo import Geo
from epymorph.geo.static import StaticGeoFileOps as F
from epymorph.geo.util import convert_to_static_geo

# TODO: add case for Mac
if system() == 'Linux':
    CACHE_PATH = Path(os.path.expanduser('~')) / '.epymorph' / 'cache'
else:
    CACHE_PATH = Path(os.path.expanduser('~')) / 'AppData' / \
        'local' / 'epymorph' / 'cache'


def cache_fetch(geo_name: str):
    """Caches all attribute data for a dynamic Geo from the library."""
    # make cache directory if needed
    if not os.path.exists(CACHE_PATH):
        os.makedirs(CACHE_PATH)

    # cache specified geo
    filepath = CACHE_PATH / F.to_archive_filename(geo_name)
    geo_load = geo_library_dynamic.get(geo_name)
    if geo_load is not None:
        geo = geo_load()
        static_geo = convert_to_static_geo(geo)
        static_geo.save(filepath)


def cache_remove(geo_name: str):
    """Removes a Geo's data from the cache."""
    filepath = CACHE_PATH / F.to_archive_filename(geo_name)
    os.remove(filepath)


def cache_list():
    """Lists the names of all currently cached Geos."""
    files = os.listdir(CACHE_PATH)
    if len(files) == 0:
        print('cache is empty')
    else:
        for file in files:
            print(F.to_geo_name(file))


def cache_clear():
    """Clears the cache of all Geo data."""
    for file in os.listdir(Path(f'{CACHE_PATH}')):
        os.remove(CACHE_PATH / file)


def swap_with_cache(dynamic_geo: Geo, geo_name: str) -> Geo:
    """Checks whether a dynamic Geo has already been cached and returns it if so."""
    file_path = CACHE_PATH / F.to_archive_filename(geo_name)
    if not os.path.exists(file_path):
        return dynamic_geo
    else:
        return F.load_from_archive(file_path)
