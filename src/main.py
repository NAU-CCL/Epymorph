import getopt
import logging
import sys
import time
from datetime import date

import numpy as np

import movement as M
import plotting as P
import simulation as S
from clock import TickDelta
from model.geo_pei import load_geo
from model.ipm_pei import PeiModel
from model.mvm_commuter import commuter_movement
from model.mvm_disperser import disperser_movement


def ruminate(plot_results: bool) -> None:
    # Simulation parameters.
    infection_duration = np.double(4)
    immunity_duration = np.double(90)
    move_control = 0.9
    theta = 0.1

    # Geo Model
    geo = load_geo()
    pop = geo.get_paramn("population", np.int_)
    hum = geo.get_paramnt("humidity", np.double)
    com = geo.get_paramnn("commuters", np.int_)

    # IPM
    ipm = PeiModel(pop, hum, D=infection_duration, L=immunity_duration)

    # Movement Model
    # (Roughly equivalent to...)
    # [move-steps: per-day=2; duration=[2/3, 1/3]]
    # [mtype: daily; days=[m,t,w,th,f,st,sn]; leave-step=1; return=0d.step2; <equation for commuters>]
    # [mtype: daily; days=[m,t,w,th,f,st,sn]; leave-step=1; return=0d.step2; <equation for dispersers>]
    mvm = M.Movement(
        # First step is day: 2/3 tau
        # Second step is night: 1/3 tau
        taus=[np.double(2/3), np.double(1/3)],
        clause=M.Sequence([
            # Main commuters: on step 0
            M.GeneralClause.byRow(
                name="Commuters",
                predicate=M.Predicates.everyday(step=0),
                returns=TickDelta(0, 1),  # returns today on step 1
                equation=commuter_movement(com.data, move_control)
            ),
            # Random dispersers: also on step 0, cumulative effect.
            M.GeneralClause.byRow(
                name="Dispersers",
                predicate=M.Predicates.everyday(step=0),
                returns=TickDelta(0, 1),  # returns today on step 1
                equation=disperser_movement(com.data, theta)
            ),
            # Return: always triggers, but only moves pops whose return time is now.
            M.Return()
        ])
    )

    # Set up the simulation...
    sim = S.Simulation(ipm, mvm, geo)

    # ... and run it.
    t0 = time.perf_counter()
    out = sim.run(
        start_date=date(2023, 1, 1),
        duration=150,
        # If you want consistent results, you can provide a seeded RNG.
        # rng=np.random.default_rng(1)
    )
    t1 = time.perf_counter()
    print(f"Simulation time: {(t1 - t0):.3f}s")

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
