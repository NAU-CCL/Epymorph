"""Logic for saving to, loading from, and managing a cache of geos on the user's hard disk."""
import os
from pathlib import Path
from typing import Callable, overload

from epymorph.cache import CACHE_PATH
from epymorph.geo.adrio.adrio import ADRIOMaker
from epymorph.geo.dynamic import DynamicGeo
from epymorph.geo.dynamic import DynamicGeoFileOps as DF
from epymorph.geo.geo import Geo
from epymorph.geo.static import StaticGeo
from epymorph.geo.static import StaticGeoFileOps as F
from epymorph.geo.util import convert_to_static_geo
from epymorph.log.messaging import dynamic_geo_messaging

AdrioMakerLibrary = dict[str, type[ADRIOMaker]]
DynamicGeoLibrary = dict[str, Callable[[], DynamicGeo]]


class GeoCacheException(Exception):
    """An exception raised when a geo cache operation fails."""


def fetch(geo_name_or_path: str,
          geo_library_dynamic: DynamicGeoLibrary,
          adrio_maker_library: AdrioMakerLibrary) -> None:
    """
    Caches all attribute data for a dynamic geo from the library or spec file at a given path.
    Raises GeoCacheException if spec not found.
    """

    # checks for geo in the library (name passed)
    if geo_name_or_path in geo_library_dynamic:
        file_path = CACHE_PATH / F.to_archive_filename(geo_name_or_path)
        geo_load = geo_library_dynamic.get(geo_name_or_path)
        if geo_load is not None:
            geo = geo_load()
            with dynamic_geo_messaging(geo):
                static_geo = convert_to_static_geo(geo)
            static_geo.save(file_path)

    # checks for geo spec at given path (path passed)
    else:
        geo_path = Path(geo_name_or_path).expanduser()
        if os.path.exists(geo_path):
            geo_name = geo_path.stem
            file_path = CACHE_PATH / F.to_archive_filename(geo_name)
            geo = DF.load_from_spec(geo_path, adrio_maker_library)
            with dynamic_geo_messaging(geo):
                static_geo = convert_to_static_geo(geo)
            static_geo.save(file_path)
        else:
            raise GeoCacheException(f'spec file at {geo_name_or_path} not found.')


def export(geo_name: str,
           geo_path: Path,
           out: str | None,
           rename: str | None,
           ignore_cache: bool,
           geo_library_dynamic: DynamicGeoLibrary,
           adrio_maker_library: AdrioMakerLibrary) -> None:
    """
    Exports a geo as a .geo.tar file to a location outside the cache.
    If uncached, geo to export is also cached.
    User can specify a destination path and new name for exported geo.
    Raises a GeoCacheException if geo not found.
    """
    # check for out path specified
    if out is not None:
        if not os.path.exists(out):
            raise GeoCacheException(f'specified output directory {out} not found.')
        else:
            out_dir = Path(out)
    else:
        out_dir = Path(os.getcwd())

    # check for geo name specified
    if rename is not None:
        geo_exp_name = rename
    else:
        geo_exp_name = geo_name

    out_path = out_dir / F.to_archive_filename(geo_exp_name)
    cache_file_path = CACHE_PATH / F.to_archive_filename(geo_name)
    cache_out_file_path = CACHE_PATH / F.to_archive_filename(geo_exp_name)

    # if cached, copy cached file
    if os.path.exists(cache_file_path):
        geo = load_from_cache(geo_name)
        if geo is not None:
            geo.save(out_path)

    # if geo uncached or spec file passed, fetch and export
    elif geo_name in geo_library_dynamic or os.path.exists(geo_path):
        geo_loader = geo_library_dynamic.get(geo_name)
        if geo_loader is not None:
            geo = geo_loader()
        else:
            geo = DF.load_from_spec(geo_path, adrio_maker_library)
        with dynamic_geo_messaging(geo):
            static_geo = convert_to_static_geo(geo)
        if not ignore_cache:
            static_geo.save(cache_out_file_path)
        static_geo.save(out_path)

    else:
        raise GeoCacheException("Geo to export not found.")


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


@overload
def load_from_cache(geo_name: str, or_else: Callable[[], StaticGeo]) -> StaticGeo:
    ...


@overload
def load_from_cache(geo_name: str) -> StaticGeo | None:
    ...


def load_from_cache(geo_name: str, or_else: Callable[[], StaticGeo] | None = None) -> StaticGeo | None:
    """
    If a geo has already been cached, load and return it.
    Otherwise, if you provide a fall-back function (`or_else`), use that to fetch a geo.
    If there is no fall-back function, `None` is returned.
    If the fallback function is used, the result will be saved to the cache.
    """
    file_path = CACHE_PATH / F.to_archive_filename(geo_name)

    if os.path.exists(file_path):
        return F.load_from_archive(file_path)

    if or_else is not None:
        geo = or_else()
        F.save_as_archive(geo, file_path)
        return geo

    return None


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
    total_size = sum((os.path.getsize(CACHE_PATH / file)
                      for file, _ in F.iterate_dir_path(CACHE_PATH)))
    return format_size(total_size)
