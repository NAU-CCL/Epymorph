"""Implements the main CLI parser."""

from argparse import ArgumentParser
from importlib.metadata import version

from epymorph.cli.cache import define_argparser as def_cache


def define_argparser() -> ArgumentParser:
    """Build a parser for all supported CLI commands."""
    # Using argparse to configure available commands and arguments.
    cli_parser = ArgumentParser(
        prog="epymorph", description="EpiMoRPH spatial meta-population modeling."
    )

    cli_parser.add_argument(
        "-V", "--version", action="version", version=version("epymorph")
    )

    # Define a set of subcommands for the main program.
    # The only requirement is that they all define a 'handler' function
    # in their defaults.
    command_parser = cli_parser.add_subparsers(
        title="commands", dest="command", required=True
    )

    def_cache(command_parser)

    return cli_parser
