import argparse
import logging

from epymorph.examples.pei_py import ruminate as pei_py_rume
from epymorph.examples.pei_spec import ruminate as pei_spec_rume
from epymorph.examples.pei_spec_n import ruminate as pei_spec_n_rume
from epymorph.movement import check_movement_spec

# This is the main entrypoint to Epymorph.
# It uses command-line options to execute one of the available sub-commands:
# - sim: runs a named, pre-configured EpiMoRPH simulation
# - check: checks the syntax of a specification file
#
# (More commands will likely be added over time.)


# "sim" subcommand
def do_sim(sim_name: str, profiling: bool, simargs: list[str]) -> int:
    """Run a named, pre-configured simulation."""

    # if we're profiling, disable logging, else normal logging
    if profiling:
        logging.basicConfig(level=logging.CRITICAL)
    else:
        logging.basicConfig(filename='debug.log', filemode='w')
        logging.getLogger('movement').setLevel(logging.DEBUG)

    # For now, all the simulations have to be coded in Python (at least in part),
    # but some day it will be possible to run a simulation from spec-files
    # loaded at runtime.

    if sim_name == 'pei_py':
        pei_py_rume(plot_results=not profiling, simargs=simargs)
        return 0  # exit code: success
    elif sim_name == 'pei_spec':
        pei_spec_rume(plot_results=not profiling, simargs=simargs)
        return 0  # exit code: success
    elif sim_name == 'pei_spec_n':
        pei_spec_n_rume(plot_results=not profiling, simargs=simargs)
        return 0  # exit code: success
    else:
        print(f"Unknown simulation: {sim_name}")
        return 1  # exit code: invalid command


# "check" subcommand
def do_check(file_path) -> int:
    """Parse and check the validity of a specification file."""
    # TODO: when there are more kinds of spec files,
    # we can switch between which we're checking based
    # on the file extension (probably).
    try:
        with open(file_path, 'r') as file:
            contents = file.read()
        try:
            check_movement_spec(contents)
            print(f"[✓] Valid specification: {file_path}")
            return 0  # exit code: success
        except Exception as e:
            print(f"[✗] Invalid specification: {file_path}")
            print(e)
            return 3  # exit code: invalid spec
    except Exception as e:
        print(f"Unable to read spec file: {e}")
        return 2  # exit code: can't read file


def main() -> None:
    # Using argparse to configure available commands and arguments.
    parser = argparse.ArgumentParser(
        prog="epymorph",
        description="EpiMoRPH spatial meta-population modeling.")
    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        required=True)

    # "sim" subcommand
    # ex: python3 -m epymorph sim pei_spec
    parser_sim = subparsers.add_parser('sim', help="run a named simulation")
    parser_sim.add_argument(
        'sim_name',
        type=str,
        help="the name of the simulation to run")
    parser_sim.add_argument(
        'simargs',
        type=str,
        nargs='*',
        help="any simulation-specific arguments"
    )
    parser_sim.add_argument(
        '-p', '--profile',
        action='store_true',
        help="(optional) include this flag to run in profiling mode")

    # "check" subcommand
    # ex: python3 -m epymorph check ./data/pei.movement
    parser_check = subparsers.add_parser(
        'check',
        help="check a specification file for validity")
    parser_check.add_argument(
        'file',
        type=str,
        help="the path to the specification file")

    args = parser.parse_args()

    exit_code = 1  # exit code: invalid command (default)
    if args.command == 'sim':
        exit_code = do_sim(args.sim_name, args.profile, args.simargs)
    elif args.command == 'check':
        exit_code = do_check(args.file)
    exit(exit_code)


if __name__ == "__main__":
    main()
