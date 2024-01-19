"""Logic for saving to, loading from, and managing a cache of geos on the user's hard disk."""
import os
from pathlib import Path

from platformdirs import user_cache_path

from epymorph.data import adrio_maker_library, geo_library_dynamic
from epymorph.geo.dynamic import DynamicGeo
from epymorph.geo.dynamic import DynamicGeoFileOps as DF
from epymorph.geo.geo import Geo
from epymorph.geo.static import StaticGeo
from epymorph.geo.static import StaticGeoFileOps as F
from epymorph.geo.util import convert_to_static_geo

CACHE_PATH = user_cache_path(appname='epymorph', ensure_exists=True)


class GeoCacheException(Exception):
    """An exception raised when a geo cache operation fails."""


def fetch(geo_name: str, geo_path=None) -> None:
    """
    Caches all attribute data for a dynamic geo from the library or spec file at a given path.
    Raises GeoCacheException if spec not found.
    """
    file_path = CACHE_PATH / F.to_archive_filename(geo_name)
    geo_load = geo_library_dynamic.get(geo_name)
    if geo_path is not None:
        geo_path = Path(geo_path)
    # checks for geo in library
    if geo_load is not None:
        geo = geo_load()
        static_geo = convert_to_static_geo(geo)
        static_geo.save(file_path)
    # checks for geo spec at given path
    elif geo_path is not None and os.path.exists(geo_path):
        geo = DF.load_from_spec(geo_path, adrio_maker_library)
        static_geo = convert_to_static_geo(geo)
        static_geo.save(file_path)
    else:
        raise GeoCacheException(f'spec file for {geo_name} not found.')


def remove(geo_name: str) -> None:
    """
    Removes a geo's data from the cache.
    Raises GeoCacheException if geo not found in cache.
    """
    file_path = CACHE_PATH / F.to_archive_filename(geo_name)
    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        msg = f'{geo_name} not found in cache, check your spelling or use the list subcommand to view all currently cached geos'
        raise GeoCacheException(msg)


def list_geos() -> list[tuple[str, int]]:
    """Return a list of all cached geos, including name and file size."""
    return [(name, os.path.getsize(CACHE_PATH / F.to_archive_filename(name)))
            for file, name in F.iterate_dir_path(CACHE_PATH)]


def clear():
    """Clears the cache of all geo data."""
    for file in F.iterate_dir_path(CACHE_PATH):
        os.remove(CACHE_PATH / file[0])


def save_to_cache(geo: Geo, geo_name: str) -> None:
    """Save a Geo to the cache (if you happen to already have it as a Geo object)."""
    match geo:
        case DynamicGeo():
            static_geo = convert_to_static_geo(geo)
        case StaticGeo():
            static_geo = geo
        case _:
            raise GeoCacheException('Unable to cache given geo.')
    file_path = CACHE_PATH / F.to_archive_filename(geo_name)
    F.save_as_archive(static_geo, file_path)


def load_from_cache(geo_name: str) -> Geo | None:
    """Checks whether a dynamic geo has already been cached and returns it if so."""
    file_path = CACHE_PATH / F.to_archive_filename(geo_name)
    if not os.path.exists(file_path):
        return None
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
    """Returns the total size of all files in the geo cache using 1024-based unit representation."""
    files = os.listdir(CACHE_PATH)
    total_size = 0
    for file in files:
        total_size += os.path.getsize(CACHE_PATH / file)

    return format_size(total_size)
