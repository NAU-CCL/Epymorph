"""The main entrypoint for epymorph: a CLI with a number of subcommands."""

import sys

from epymorph.cli.main import define_argparser


def main() -> None:
    """Execute epymorph's CLI entrypoint."""
    args = define_argparser().parse_args()
    exit_code = args.handler(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
