"""
Implements the `cache` subcommands executed from __main__.
"""

from argparse import _SubParsersAction

from epymorph.cache import (
    CACHE_PATH,
    Directory,
    FileError,
    cache_inventory,
    cache_remove_confirmation,
    format_file_size,
)


def define_argparser(command_parser: _SubParsersAction):
    """Define `cache` subcommand."""
    p = command_parser.add_parser("cache", help="manage epymorph's file cache")

    sp = p.add_subparsers(title="cache_commands", dest="cache_commands", required=True)

    list_command = sp.add_parser("list", help="list the contents of the cache")
    list_command.set_defaults(handler=lambda args: handle_list())

    remove_command = sp.add_parser(
        "remove", help="remove a file or folder from the cache"
    )
    remove_command.add_argument(
        "path", type=str, help="the relative path to a file or folder in the cache"
    )
    remove_command.set_defaults(handler=lambda args: handle_remove(path=args.path))


def handle_list() -> int:
    """CLI command handler: cache list."""

    def print_folders_in(directory: Directory, indent: str = "  "):
        child_dirs = (d for d in directory.children if isinstance(d, Directory))
        for x in sorted(child_dirs, key=lambda x: x.name):
            print(f"{indent}- {x.name} ({format_file_size(x.size)})")
            print_folders_in(x, indent + "  ")

    cache = cache_inventory()
    print(f"epymorph cache is using {format_file_size(cache.size)} ({CACHE_PATH})")
    print_folders_in(cache)
    return 0  # exit code: success


def handle_remove(path: str) -> int:
    """CLI command handler: remove a file or folder from the cache."""
    try:
        to_remove, confirm_remove = cache_remove_confirmation(path)
        if to_remove.is_dir():
            print(f"This will delete all cache entries at {to_remove}")
        else:
            print(f"This will delete the cached file {to_remove}")
        response = input("Are you sure? [y/N]: ")
        if response.lower() in ("y", "yes"):
            confirm_remove()
        return 0  # exit code: success
    except FileError as e:
        print(f"Error: {e}")
        return 1  # exit code: failed
