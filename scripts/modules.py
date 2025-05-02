"""
Auto-generate mkdocs files for every epymorph module.
I think there are other more typical ways to do this, but just prototyping here.
"""

# ruff: noqa: T201
import importlib
import importlib.util
import pkgutil
import textwrap
from pathlib import Path


def discover_modules(package_name: str) -> list[str]:
    spec = importlib.util.find_spec(package_name)
    if spec is None or spec.submodule_search_locations is None:
        raise Exception(f"{package_name} is not a package")

    return [
        module_info.name
        for module_info in pkgutil.walk_packages(
            spec.submodule_search_locations,
            prefix=package_name + ".",
        )
        if not module_info.ispkg
    ]


def format_file(module: str) -> tuple[str, str]:
    filename = f"{module.removeprefix('epymorph.').replace('.', '_')}.md"
    content = f"""\
    ---
    title: "{module.removeprefix("epymorph")}"
    ---

    ::: {module}
    """
    return filename, textwrap.dedent(content)


if __name__ == "__main__":
    docs_dir = Path("docs")
    modules = discover_modules("epymorph")
    for m in modules:
        name, content = format_file(m)
        print(f"Writing {name}")
        with (docs_dir / name).open("w") as file:
            file.write(content)
