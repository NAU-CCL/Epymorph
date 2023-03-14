import getopt
import logging
import sys
import time
from datetime import date

import numpy as np

import movement as M
import pei
import plotting as P
import simulation as S
from clock import Tick


def ruminate(plot_results: bool) -> None:
    geo = pei.loadGeo()
    pop = geo.get_paramn("population", np.int_)
    hum = geo.get_paramnt("humidity", np.double)
    # def hum(node_idx: int, tick: Tick) -> np.double: return np.double(0.01003)
    com = geo.get_paramnn("commuters", np.int_)

    ipm = pei.PeiModel(pop, hum, D=np.double(4), L=np.double(90))

    # mvm = M.Movement([
    #     M.Step(np.double(1), M.Noop())
    # ])

    mvm = M.Movement([
        M.Step(
            np.double(2/3),
            # M.FixedCommuteMatrix(duration=1, commuters=np.full(
            #     (geo.num_nodes, geo.num_nodes), 1000))
            M.Sequence([
                M.StochasticCommuteMatrix(
                    duration=1, commuters=com.data, move_control=1.0),
                M.RandomDispersersMatrix(
                    duration=1, commuters=com.data, theta=0.1)
            ])
        ),
        M.Step(
            np.double(1/3),
            M.Return()
        )
    ])

    sim = S.Simulation(ipm, mvm, geo)

    t0 = time.perf_counter()
    out = sim.run(date(2023, 1, 1), 150)
    t1 = time.perf_counter()
    print(f"Simulation time: {(t1 - t0):.2f}s")

    # Output
    if plot_results:
        # Plot infections (per-100k):
        # P.plot_events(out, 0, lambda i, values: 100_000 * values / pop(i))

        # Plot infections (not scaled):
        P.plot_events(out, 0, lambda _, values: values)

        # Plot prevalence for each compartment in the first population (log scale):
        # P.plot_pop_prevalence(out, 0, lambda values: np.log(values))

        # Plot prevalence for each compartment in the first population (not scaled):
        # P.plot_pop_prevalence(out, 0, identity)


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
