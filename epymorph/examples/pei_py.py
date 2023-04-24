import time
from datetime import date

import numpy as np

import epymorph.plotting as P
import epymorph.simulation as S
from epymorph.model.geo_pei import load_geo as load_pei_geo
from epymorph.model.ipm_pei import PeiModelBuilder
from epymorph.model.mvm_pei import load_mvm as load_pei_mvm


# An example script which runs a Pei-like simulation using modules encoded in python.
def ruminate(plot_results: bool) -> None:
    # Set up the simulation...
    sim = S.Simulation(
        geo=load_pei_geo(),
        ipmBuilder=PeiModelBuilder(),
        mvmBuilder=load_pei_mvm()
    )

    # ... and run it.
    t0 = time.perf_counter()
    out = sim.run(
        param={
            'theta': 0.1,
            'move_control': 0.9,
            'infection_duration': np.double(4),
            'immunity_duration': np.double(90),
            'infection_seed_loc': 0,
            'infection_seed_size': 10_000
        },
        start_date=date(2023, 1, 1),
        duration=150,
        # If you want consistent results, you can provide a seeded RNG.
        rng=np.random.default_rng(1)
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


if __name__ == "__main__":
    ruminate(True)
