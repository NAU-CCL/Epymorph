import getopt
import logging
import sys

from epymorph.examples.pei_py import ruminate as pei_py_rume


def configure_logging(profiling: bool) -> None:
    if profiling:
        logging.basicConfig(level=logging.CRITICAL)
    else:
        logging.basicConfig(filename='debug.log', filemode='w')
        logging.getLogger('movement').setLevel(logging.DEBUG)


def main(argv: list[str]) -> None:
    # Argument processing
    profiling = False
    sim = None

    opts, args = getopt.getopt(argv, 's:', ['sim=', 'profile'])
    for opt, value in opts:
        if opt in ('-s', '--sim'):
            sim = value
        elif opt == '--profile':
            profiling = True

    configure_logging(profiling)

    if sim == 'pei_py':
        pei_py_rume(plot_results=not profiling)
    elif sim == None:
        print("Please choose a simulation to run.")
    else:
        print(f"Unknown simulation: {sim}")


if __name__ == "__main__":
    main(sys.argv[1:])
