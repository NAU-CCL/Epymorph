"""Implements the `cache` CLI subcommands."""
# ruff: noqa: T201

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
    cmd = p.add_subparsers(title="cache_commands", dest="cache_commands", required=True)

    list_cmd = cmd.add_parser("list", help="list the contents of the cache")
    list_cmd.set_defaults(handler=lambda args: handle_list())

    remove_cmd = cmd.add_parser("remove", help="remove a file or folder from the cache")
    remove_cmd.add_argument(
        "path",
        type=str,
        help=(
            "the relative path to a file or folder in the cache, "
            "or omit to purge the cache entirely"
        ),
        nargs="?",
        default=".",
    )
    remove_cmd.set_defaults(handler=lambda args: handle_remove(path=args.path))

    dir_cmd = cmd.add_parser("dir", help="output the absolute path of the cache folder")
    dir_cmd.set_defaults(handler=lambda args: handle_dir())


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
        if path == ".":
            print("This will delete all cache entries")
        elif to_remove.is_dir():
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


def handle_dir() -> int:
    """CLI command handler: print the path to the cache directory."""
    print(CACHE_PATH)
    return 0
