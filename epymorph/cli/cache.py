"""
Implements the `cache` subcommands executed from __main__.
"""
import os
from argparse import _SubParsersAction
from pathlib import Path

from epymorph.data import geo_library, geo_library_dynamic
from epymorph.geo import cache
from epymorph.geo.static import StaticGeoFileOps as F


def define_argparser(command_parser: _SubParsersAction):
    """
    Define `cache` subcommand.
    ex: `epymorph cache <geo name>`
    """
    p = command_parser.add_parser(
        'cache',
        help='cache geos and access geo cache information')

    sp = p.add_subparsers(
        title='cache_commands',
        dest='cache_commands',
        required=True)

    fetch_command = sp.add_parser(
        'fetch',
        help='fetch and cache data for a geo')
    fetch_command.add_argument(
        'geo',
        type=str,
        help='the name of the geo to fetch; must include a geo path if not already in the library')
    fetch_command.add_argument(
        '-f', '--force',
        action='store_true',
        help='(optional) include this flag to force an override of previously cached data')
    fetch_command.set_defaults(handler=lambda args: fetch(
        geo_name_or_path=args.geo,
        force=args.force
    ))

    remove_command = sp.add_parser(
        'remove',
        help="remove a geo's data from the cache")
    remove_command.add_argument(
        'geo',
        type=str,
        help='the name of a geo from the library')
    remove_command.set_defaults(handler=lambda args: remove(
        geo_name=args.geo
    ))

    list_command = sp.add_parser(
        'list',
        help='list the names of all currently cached geos')
    list_command.set_defaults(handler=lambda args: print_geos())

    clear_command = sp.add_parser(
        'clear',
        help='clear the cache')
    clear_command.set_defaults(handler=lambda args: clear())


# Exit codes:
# - 0 success
# - 1 geo not found
# - 2 empty cache


def fetch(geo_name_or_path: str, force: bool) -> int:
    """CLI command handler: cache dynamic geo data."""

    # split geo name and path
    if geo_name_or_path in geo_library_dynamic:
        geo_name = geo_name_or_path
        geo_path = None
    elif os.path.exists(Path(geo_name_or_path).expanduser()):
        geo_path = Path(geo_name_or_path).expanduser()
        geo_name = geo_path.stem
    else:
        raise cache.GeoCacheException("Specified geo not found.")

    # cache geo according to information passed
    filepath = cache.CACHE_PATH / F.to_archive_filename(geo_name)
    if geo_path is not None and geo_name in geo_library:
        msg = f"A geo named {geo_name} is already present in the library. Please use the existing geo or change the file name."
        raise cache.GeoCacheException(msg)
    choice = 'y'
    if os.path.exists(filepath) and not force:
        choice = input(f'{geo_name} is already cached, overwrite? [y/n] ')
    if force or choice == 'y':
        try:
            cache.fetch(geo_name_or_path)
            print("geo sucessfully cached.")
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
