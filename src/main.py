import getopt
import logging
import sys
import time
from datetime import date

import numpy as np

import pei
import plotting as P
import simulation as S
from util import identity


def ruminate(plot_results: bool) -> None:
    geo = pei.load_geo()
    pop = geo.get_paramn("population", np.int_)
    hum = geo.get_paramnt("humidity", np.double)
    com = geo.get_paramnn("commuters", np.int_)

    ipm = pei.PeiModel(pop, hum, D=np.double(4), L=np.double(90))

    # Example: no movement, single tau step.
    # mvm = M.Movement(
    #     taus=[np.double(1)],
    #     clause=M.Noop()
    # )

    mvm = pei.build_movement(
        commuters=com.data,
        move_control=0.9,
        theta=0.1
    )

    sim = S.Simulation(ipm, mvm, geo)

    t0 = time.perf_counter()
    out = sim.run(
        start_date=date(2023, 1, 1),
        duration=150,
        # If you want consistent results, you can provide a seeded RNG.
        # rng=np.random.default_rng(1)
    )
    t1 = time.perf_counter()
    print(f"Simulation time: {(t1 - t0):.2f}s")

    # Output
    if plot_results:
        # Plot infections (per-100k):
        # P.plot_events(
        #     out,
        #     event_idx=0,
        #     labels={
        #         'title': 'Infection incidence',
        #         'x_label': 'days',
        #         'y_label': 'events per 100k population'
        #     },
        #     scaling=lambda i, values: 100_000 * values / pop(i)
        # )

        # Plot infections (not scaled):
        P.plot_events(
            out,
            event_idx=0,
            labels={
                'title': 'Infection incidence',
                'x_label': 'days',
                'y_label': 'events'
            },
            scaling=lambda _, values: values
        )

        # Plot prevalence for each compartment in the first population (log scale):
        # P.plot_pop_prevalence(
        #     out,
        #     pop_idx=0,
        #     labels={
        #         'title': 'Prevalence in FL',
        #         'x_label': 'days',
        #         'y_label': 'log(persons)'
        #     },
        #     scaling=lambda values: np.log(values)
        # )

        # Plot prevalence for each compartment in the first population (not scaled):
        # P.plot_pop_prevalence(
        #     out,
        #     pop_idx=0,
        #     labels={
        #         'title': 'Prevalence in FL',
        #         'x_label': 'days',
        #         'y_label': 'persons'
        #     },
        #     scaling=identity
        # )


def main(argv: list[str]) -> None:
    # Argument processing
    profiling = False
    opts, args = getopt.getopt(argv, '', ['profile'])
    for opt, value in opts:
        if opt == "--profile":
            profiling = True

    # Logging setup
    if profiling:
        logging.basicConfig(level=logging.CRITICAL)
    else:
        logging.basicConfig(filename='debug.log', filemode='w')
        logging.getLogger('movement').setLevel(logging.DEBUG)

    ruminate(plot_results=not profiling)


if __name__ == "__main__":
    main(sys.argv[1:])
