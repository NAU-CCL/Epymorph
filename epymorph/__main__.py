import argparse

from epymorph.movement import check_movement_spec
from epymorph.run import run as epymorph_run
from epymorph.verify import verify as epymorph_verify

# This is the main entrypoint to Epymorph.
# It uses command-line options to execute one of the available sub-commands:
# - sim: runs a named, pre-configured EpiMoRPH simulation
# - check: checks the syntax of a specification file
#
# (More commands will likely be added over time.)


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

    # "run" subcommand
    # ex: python3 -m epymorph run --ipm pei --mm pei --geo pei
    parser_run = subparsers.add_parser(
        'run', help="run a simulation from library models")
    parser_run.add_argument(
        '--ipm',
        type=str,
        required=True,
        help="the name of an IPM from the library")
    parser_run.add_argument(
        '--mm',
        type=str,
        required=True,
        help="the name of an MM from the library")
    parser_run.add_argument(
        '--geo',
        type=str,
        required=True,
        help="the name of a Geo from the library")
    parser_run.add_argument(
        '--start_date',
        type=str,
        required=True,
        help="the start date of the simulation in ISO format, e.g.: 2023-01-27")
    parser_run.add_argument(
        '--duration',
        type=str,
        required=True,
        help="the timespan of the simulation, e.g.: 30d")
    parser_run.add_argument(
        '--params',
        type=str,
        required=True,  # TODO: make this optional and use default values
        help="the path to a params file")
    parser_run.add_argument(
        '--out',
        help="(optional) path to an output file to save the simulated prevalence data; specify either a .csv or .npz file")
    parser_run.add_argument(
        '--chart',
        help="(optional) ID for chart to draw; \"e0\" for event incidence 0; \"p2\" for pop prevalence 2; etc. (this is a temporary feature in lieu of better output handling)")
    parser_run.add_argument(
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

    # "verify" subcommand
    # ex: python3 -m epymorph verify ./output.csv
    parser_verify = subparsers.add_parser(
        'verify',
        help="check output file for data consistency")
    parser_verify.add_argument(
        'file',
        type=str,
        help="the path to the output file")

    args = parser.parse_args()

    exit_code = 1  # exit code: invalid command (default)
    if args.command == 'run':
        exit_code = epymorph_run(
            args.ipm,
            args.mm,
            args.geo,
            args.start_date,
            args.duration,
            args.params,
            args.out,
            args.chart,
            args.profile
        )
    elif args.command == 'check':
        exit_code = do_check(args.file)
    elif args.command == 'verify':
        exit_code = epymorph_verify(args.file)
    exit(exit_code)


if __name__ == "__main__":
    main()
