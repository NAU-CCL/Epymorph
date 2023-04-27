import getopt
import logging
import sys

from epymorph.examples.pei_py import ruminate as pei_py_rume
from epymorph.examples.pei_spec import ruminate as pei_spec_rume
from epymorph.movement import check_movement_spec


def configure_logging(profiling: bool) -> None:
    if profiling:
        logging.basicConfig(level=logging.CRITICAL)
    else:
        logging.basicConfig(filename='debug.log', filemode='w')
        logging.getLogger('movement').setLevel(logging.DEBUG)


def do_check(file_path) -> None:
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
        except Exception as e:
            print(f"[✗] Invalid specification: {file_path}")
            print(e)
    except Exception as e:
        print(f"Unable to read spec file: {e}")


def main(argv: list[str]) -> None:
    # Argument processing
    profiling = False
    sim: str | None = None
    check: str | None = None

    opts, args = getopt.getopt(argv, 's:c:', ['sim=', 'check=', 'profile'])
    for opt, value in opts:
        if opt in ('-s', '--sim'):
            sim = value
        elif opt in ('-c', '--check'):
            check = value
        elif opt == '--profile':
            profiling = True

    configure_logging(profiling)

    if check is not None:
        do_check(check)
    elif sim == 'pei_py':
        pei_py_rume(plot_results=not profiling)
    elif sim == 'pei_spec':
        pei_spec_rume(plot_results=not profiling)
    elif sim is None:
        print("Please choose a simulation to run.")
    else:
        print(f"Unknown simulation: {sim}")


if __name__ == "__main__":
    main(sys.argv[1:])
