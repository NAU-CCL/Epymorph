import argparse
import logging

from epymorph.examples.pei_py import ruminate as pei_py_rume
from epymorph.examples.pei_spec import ruminate as pei_spec_rume
from epymorph.movement import check_movement_spec


def configure_logging(profiling: bool) -> None:
    if profiling:
        logging.basicConfig(level=logging.CRITICAL)
    else:
        logging.basicConfig(filename='debug.log', filemode='w')
        logging.getLogger('movement').setLevel(logging.DEBUG)


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
            return 2  # exit code: invalid spec
    except Exception as e:
        print(f"Unable to read spec file: {e}")
        return 1  # exit code: can't read file


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="epymorph",
        description="EpiMoRPH spatial meta-population modeling.")
    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        required=True)

    # sim subcommand
    parser_sim = subparsers.add_parser('sim', help="run a named simulation")
    parser_sim.add_argument(
        'sim_name',
        type=str,
        help="the name of the simulation to run")
    parser_sim.add_argument(
        '-p', '--profile',
        action='store_true',
        help="(optional) include this flag to run in profiling mode")

    # check subcommand
    parser_check = subparsers.add_parser(
        'check',
        help="check a specification file for validity")
    parser_check.add_argument(
        'file',
        type=str,
        help="the path to the specification file")

    args = parser.parse_args()

    if args.command == 'sim':
        configure_logging(args.profile)
        if args.sim_name == 'pei_py':
            pei_py_rume(plot_results=not args.profile)
        elif args.sim_name == 'pei_spec':
            pei_spec_rume(plot_results=not args.profile)
        else:
            print(f"Unknown simulation: {args.sim_name}")

    elif args.command == 'check':
        exit_code = do_check(args.file)
        exit(exit_code)


if __name__ == "__main__":
    main()
