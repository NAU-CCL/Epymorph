import argparse

from epymorph.movement import check_movement_spec

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
    if args.command == 'check':
        exit_code = do_check(args.file)
    exit(exit_code)


if __name__ == "__main__":
    main()
