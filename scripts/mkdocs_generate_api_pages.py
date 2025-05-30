#!/usr/bin/env -S uv run --script
"""
Auto-generate mkdocs files for all of epymorph's modules (with some exclusions).
This script is configured to run during mkdocs generation via the mkdocs-gen-files
plugin.
"""

import fnmatch
import importlib
import importlib.util
import logging
import pkgutil
import sys
import textwrap
from contextlib import contextmanager
from pathlib import Path

import mkdocs_gen_files

log = logging.getLogger("mkdocs")

PACKAGE = "epymorph"
EXCLUDES = [
    "epymorph.__main__",
    "epymorph.sympy_shim",
    "epymorph.cli.*",
]
"""
Modules to exclude from the documentation, expressed as glob-style patterns matched
against the module's fully qualified dotted path.
"""


def discover_modules(package_name: str, excludes: list[str]) -> list[str]:
    """Find all of the package's modules if they're not in the exclusion list."""
    spec = importlib.util.find_spec(package_name)
    if spec is None or spec.submodule_search_locations is None:
        raise Exception(f"Unable to find package '{package_name}'")

    def is_excluded(module_name: str) -> bool:
        return any(fnmatch.fnmatch(module_name, x) for x in excludes)

    return [
        module_info.name
        for module_info in pkgutil.walk_packages(
            spec.submodule_search_locations,
            prefix=package_name + ".",
        )
        if not module_info.ispkg
        if not is_excluded(module_info.name)
    ]


def write_module_stubs(open_file) -> int:
    """Generate docs stub files for all modules. Return number of files generated."""

    def format_docs_stub(module: str) -> tuple[str, str]:
        """
        Format a stub file that mkdocstrings will use to generate API documentation
        for the given module. Return the file name and content as a tuple.
        """
        filename = (
            f"{module.removeprefix(PACKAGE).removeprefix('.').replace('.', '_')}.md"
        )
        content = f"""\
        ---
        title: "{module.removeprefix(PACKAGE)}"
        ---

        # {module}

        ::: {module}
        """
        return filename, textwrap.dedent(content)

    n = 0
    docs_dir = Path("docs")
    for m in discover_modules(PACKAGE, EXCLUDES):
        filename, content = format_docs_stub(m)

        # Skip if a matching file already exists in the docs dir.
        # This allows you to override document page generation per module.
        if (docs_dir / filename).exists():
            log.info(f"Found override doc file for '{filename}'; skipping generation.")
            continue

        # Write the document.
        with open_file(filename, "w") as file:
            file.write(content)
        n += 1
    return n


if __name__ == "<run_path>":
    # Running from mkdocs
    # > uv run mkdocs serve
    # --or--
    # > uv run mkdocs build
    log.info("Auto-generating doc files (scripts/mkdocs_generate_api_pages.py)...")
    n = write_module_stubs(mkdocs_gen_files.open)
    log.info(f"Generated {n} files.")


if __name__ == "__main__":
    # Running directly (useful for debugging)

    # list mode: just list discovered modules
    # > scripts/mkdocs_generate_api_pages.py list
    if len(sys.argv) > 1 and sys.argv[1] == "list":
        for module in discover_modules(PACKAGE, EXCLUDES):
            print(module)  # noqa: T201
        sys.exit(0)

    # write mode: write stub files to `mkdocs_debug` directory
    # > scripts/mkdocs_generate_api_pages.py
    out_dir = Path("mkdocs_debug")
    out_dir.mkdir(exist_ok=True)

    @contextmanager
    def open_file(name: str, mode: str):
        with (out_dir / name).open(mode=mode) as file:
            print(f"Writing {name}")  # noqa: T201
            yield file

    write_module_stubs(open_file)
