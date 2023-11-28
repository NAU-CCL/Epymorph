"""
Implements the `validate` subcommand executed from __main__.
"""
from typing import Any, Callable

from epymorph.movement.parser import parse_movement_spec


def validate_spec_file(file_path: str) -> int:
    """CLI command handler: check the validity of a specification file."""

    # Exit codes:
    # - 0 success
    # - 2 can't read file
    # - 3 spec is invalid

    validation_func: Callable[[str], Any]
    if file_path.endswith(".movement"):
        validation_func = parse_movement_spec
    # elif file_path.endswith(".geo"):
    #     ...
    # elif file_path.endswith(".ipm");
    #     ...
    else:
        print(f"[✗] Unknown spec file type: {file_path}")
        return 1  # exit code: invalid spec file

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            contents = file.read()
        validation_func(contents)
        print(f"[✓] Valid specification: {file_path}")
        return 0  # exit code: success
    except OSError as e:
        print(e)
        print(f"[✗] Unable to open file: {file_path}")
        return 2  # exit code: can't read file
    except Exception as e:
        print(e)
        print(f"[✗] Invalid specification: {file_path}")
        return 3  # exit code: invalid spec
