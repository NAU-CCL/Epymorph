"""The main entrypoint for epymorph: a CLI with a number of subcommands."""
import sys
from argparse import ArgumentParser
from importlib.metadata import version

from epymorph.cache import cache_clear as handle_clear
from epymorph.cache import cache_fetch as handle_fetch
from epymorph.cache import cache_list as handle_list
from epymorph.cache import cache_remove as handle_remove
from epymorph.prepare import prepare_run_toml as handle_prepare
from epymorph.run import run as handle_run
from epymorph.validate import validate_spec_file as handle_validate
from epymorph.verify import verify as handle_verify


def build_cli() -> ArgumentParser:
    """Builds a parser for all supported CLI commands."""
    # Using argparse to configure available commands and arguments.
    cli_parser = ArgumentParser(
        prog="epymorph",
        description="EpiMoRPH spatial meta-population modeling.")

    cli_parser.add_argument('-V', '--version', action='version',
                            version=version('epymorph'))

    # Define a set of subcommands for the main program.
    # Each is defined in an immediately-executed function below,
    # the only requirement is that they all define a 'handler' function
    # in their defaults.
    command_parser = cli_parser.add_subparsers(
        title="commands",
        dest="command",
        required=True)

    # define "run" subcommand
    # ex: python3 -m epymorph run ./scratch/params.toml --chart e0
    def define_run():
        p = command_parser.add_parser(
            'run', help="run a simulation from library models")
        p.add_argument(
            'input',
            help="the path to an input toml file")
        p.add_argument(
            '-e', '--engine',
            help="(optional) the id of a runtime engine to use")
        p.add_argument(
            '-o', '--out',
            help="(optional) path to an output file to save the simulated prevalence data; specify either a .csv or .npz file")
        p.add_argument(
            '-c', '--chart',
            help="(optional) ID for chart to draw; \"e0\" for event incidence 0; \"p2\" for pop prevalence 2; etc. (this is a temporary feature in lieu of better output handling)")
        p.add_argument(
            '-p', '--profile',
            action='store_true',
            help="(optional) include this flag to run in profiling mode")
        p.add_argument(
            '-i', '--ignore_cache',
            help='(optional) include this flag to run the simulation without utilizing the Geo cache.'
        )

        def handler(args):
            return handle_run(args.input, args.engine, args.out, args.chart, args.profile, args.ignore_cache)
        p.set_defaults(handler=handler)
    define_run()

    # define "prepare" subcommand
    # ex: python3 -m epymorph prepare ./scratch/params.toml
    def define_prepare():
        p = command_parser.add_parser(
            'prepare', help="prepare an input toml file for the run command")
        p.add_argument(
            'file',
            help="the path at which to save the file")
        p.add_argument(
            '--ipm',
            type=str,
            help="(optional) the name of an IPM from the library")
        p.add_argument(
            '--mm',
            type=str,
            help="(optional) the name of an MM from the library")
        p.add_argument(
            '--geo',
            type=str,
            help="(optional) the name of a Geo from the library")

        def handler(args):
            return handle_prepare(args.file, args.ipm, args.mm, args.geo)
        p.set_defaults(handler=handler)
    define_prepare()

    # define "cache" subcommand
    # ex: python3 -m epymorph cache <geo name>
    def define_cache():
        p = command_parser.add_parser(
            'cache',
            help='cache Geos and access Geo cache information')
        sp = p.add_subparsers(
            title='cache_commands',
            dest='cache_commands',
            required=True)
        fetch = sp.add_parser(
            'fetch',
            help='fetch and cache data for a Geo')
        fetch.add_argument(
            'geo',
            type=str,
            help='the name of a geo from the library')
        fetch.add_argument(
            '-f', '--force',
            action='store_true',
            help='(optional) include this flag to force an override of previously cached data')
        remove = sp.add_parser(
            'remove',
            help='remove a Geo\'s data from the cache')
        remove.add_argument(
            'geo',
            type=str,
            help='the name of a Geo from the library')
        cache_list = sp.add_parser(
            'list',
            help='list the names of all currently cached Geos')
        clear = sp.add_parser(
            'clear',
            help='clear the cache')

        def fetch_handler(args):
            return handle_fetch(args.geo, args.force)

        def remove_handler(args):
            return handle_remove(args.geo)

        def list_handler(args):
            return handle_list()

        def clear_handler(args):
            return handle_clear()

        fetch.set_defaults(handler=fetch_handler)
        remove.set_defaults(handler=remove_handler)
        cache_list.set_defaults(handler=list_handler)
        clear.set_defaults(handler=clear_handler)
    define_cache()

    # define "check" subcommand
    # ex: python3 -m epymorph check ./epymorph/data/mm/pei.movement
    def define_check():
        p = command_parser.add_parser(
            'check',
            help="check a specification file for validity")
        p.add_argument(
            'file',
            type=str,
            help="the path to the specification file")

        def handler(args):
            return handle_validate(args.file)
        p.set_defaults(handler=handler)
    define_check()

    # define "verify" subcommand
    # ex: python3 -m epymorph verify ./output.csv
    def define_verify():
        p = command_parser.add_parser(
            'verify',
            help="check output file for data consistency")
        p.add_argument(
            'file',
            type=str,
            help="the path to the output file")

        def handler(args):
            return handle_verify(args.file)
        p.set_defaults(handler=handler)
    define_verify()

    return cli_parser


def main() -> None:
    """The main entrypoint for epymorph."""
    args = build_cli().parse_args()
    exit_code = args.handler(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
