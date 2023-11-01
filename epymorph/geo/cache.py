import os

from platformdirs import user_cache_path

from epymorph.data import geo_library_dynamic
from epymorph.geo.geo import Geo
from epymorph.geo.static import StaticGeoFileOps as F
from epymorph.geo.util import convert_to_static_geo

CACHE_PATH = user_cache_path(appname='epymorph', ensure_exists=True)


def fetch(geo_name: str) -> bool:
    """
    Caches all attribute data for a dynamic geo from the library.
    Returns true if geo fetched or false if spec not found.
    """
    filepath = CACHE_PATH / F.to_archive_filename(geo_name)
    geo_load = geo_library_dynamic.get(geo_name)
    if geo_load is not None:
        geo = geo_load()
        static_geo = convert_to_static_geo(geo)
        static_geo.save(filepath)
        return True
    else:
        return False


def remove(geo_name: str) -> bool:
    """
    Removes a geo's data from the cache.
    Returns true if geo removed or false if not found.
    """
    filepath = CACHE_PATH / F.to_archive_filename(geo_name)
    if not os.path.exists(filepath):
        return False
    else:
        os.remove(filepath)
        return True


def list():
    """Lists the names of all currently cached geos."""
    files = os.listdir(CACHE_PATH)
    for file in files:
        print(
            f"* {F.to_geo_name(file)} ({format_size(os.path.getsize(CACHE_PATH / file))})")


def clear():
    """Clears the cache of all geo data."""
    for file in os.listdir(CACHE_PATH):
        os.remove(CACHE_PATH / file)


def swap_with_cache(dynamic_geo: Geo, geo_name: str) -> Geo:
    """Checks whether a dynamic geo has already been cached and returns it if so."""
    file_path = CACHE_PATH / F.to_archive_filename(geo_name)
    if not os.path.exists(file_path):
        return dynamic_geo
    else:
        return F.load_from_archive(file_path)


def format_size(size: int) -> str:
    """
    Given a file size in bytes, produce a 1024-based unit representation
    with the decimal in a consistent position, and padded with spaces as necessary.
    """
    if abs(size) < 1024:
        return f"{size:3d}.  "

    fnum = float(size)
    magnitude = 0
    while abs(fnum) > 1024:
        magnitude += 1
        fnum = int(fnum / 100.0) / 10.0
    suffix = [' B', ' kiB', ' MiB', ' GiB'][magnitude]
    return f"{fnum:.1f}{suffix}"


def get_total_size() -> str:
    files = os.listdir(CACHE_PATH)
    total_size = 0
    for file in files:
        total_size += os.path.getsize(CACHE_PATH / file)

    return format_size(total_size)
