#!/usr/bin/env -S uv run --script
"""Extract the changelog for a specified version."""

import argparse
import re
import sys
from pathlib import Path


def extract_section(changelog: str, version: str) -> str:
    # Find the start of the section for the given version.
    open_header = list(
        re.finditer(
            rf"^## \[{re.escape(version)}\] - .+$",
            changelog,
            flags=re.MULTILINE,
        )
    )

    if not open_header:
        raise ValueError(f"No changelog section found for version {version!r}.")
    if len(open_header) > 1:
        raise ValueError(f"Multiple changelog sections found for version {version!r}.")

    start = open_header[0].end()

    # Find the next section heading or the reference links (footer).
    next_boundary = re.search(
        r"^## \[.+\] - .+$|^\[[^\]]+\]:\s+\S",
        changelog[start:],
        re.MULTILINE,
    )

    end = start + next_boundary.start() if next_boundary else len(changelog)

    # Strip whitespace.
    section = changelog[start:end].strip()

    # Promote all headings by one level. (e.g., "### Foo" -> "## Foo")
    return re.sub(
        r"^(#{2,6})(?= )",
        lambda match: match.group(1)[1:],
        section,
        flags=re.MULTILINE,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a release's changelog section to standard output.",
    )
    parser.add_argument(
        "version",
        help="Release version, optionally prefixed with 'v'.",
    )
    parser.add_argument(
        "changelog",
        type=Path,
        nargs="?",
        default=Path("CHANGELOG.md"),
        help="Changelog file to read (default: CHANGELOG.md).",
    )
    args = parser.parse_args()

    try:
        version = args.version.removeprefix("v")  # normalize version string
        changelog = args.changelog.read_text(encoding="utf-8")
        notes = extract_section(changelog, version)
    except (OSError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
    else:
        sys.stdout.write(notes + "\n")


if __name__ == "__main__":
    main()
