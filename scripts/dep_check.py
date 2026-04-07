#!/usr/bin/env -S uv run --script
# ruff: noqa: T201

"""
A utility script for checking that all external dependencies used in the code are
declared in pyproject.toml, and that all declared dependencies are actually used
in the code. Exits with a non-zero code if either condition is violated.
"""

import ast
import re
import sys
import tomllib
from pathlib import Path
from typing import Iterable

PACKAGE_NAME = re.compile(r"^[A-Za-z0-9_\-]+")


def extract_imports(path: Path) -> Iterable[str]:
    """Returns just the top-level module names imported in a Python file"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                yield node.module.split(".")[0]


def classify_modules(modules: set[str]) -> tuple[set[str], set[str]]:
    """Split modules into built-ins vs external"""
    stdlib = sys.stdlib_module_names
    builtins = {m for m in modules if m in stdlib}
    external = modules - builtins
    return builtins, external


def extract_dependencies(pyproject_path: Path) -> Iterable[str]:
    """Load declared runtime dependencies from pyproject.toml"""
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    for dep in data.get("project", {}).get("dependencies", []):
        if match := PACKAGE_NAME.match(dep):
            yield match.group(0)


def main() -> None:
    project = "epymorph"
    source_root = Path.cwd() / project
    external_imports = set(
        module
        for file in source_root.rglob("*.py")
        for module in extract_imports(file)
        if not module == project  # we don't use relative imports
    )

    builtins, external = classify_modules(external_imports)
    declared_deps = set(extract_dependencies(Path.cwd() / "pyproject.toml"))
    used_but_not_declared = {m for m in external if m.lower() not in declared_deps}
    declared_but_not_used = {d for d in declared_deps if d not in external}

    verbose = "--verbose" in sys.argv

    if verbose:
        print("=== Built-in Modules ===")
        for m in sorted(builtins):
            print(m)
        print("" if builtins else "(none)\n")

        print("=== External Modules ===")
        for m in sorted(external):
            print(m)
        print("" if external else "(none)\n")

    if verbose or used_but_not_declared:
        print("=== External Modules Used but NOT Declared ===")
        for m in sorted(used_but_not_declared):
            print(m)
        print("" if used_but_not_declared else "(none)\n")

    if verbose or declared_but_not_used:
        print("=== Dependencies Declared but NOT Used ===")
        for d in sorted(declared_but_not_used):
            print(d)
        print("" if declared_but_not_used else "(none)\n")

    if used_but_not_declared or declared_but_not_used:
        print("FAIL: dependency issues found. See above for details.")
        sys.exit(1)
    else:
        print("PASS: all dependencies accounted for.")
        sys.exit(0)


if __name__ == "__main__":
    main()
