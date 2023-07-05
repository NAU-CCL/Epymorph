from datetime import date

from epymorph.data import geo_library, ipm_library, mm_library
from epymorph.simulation import Simulation


def main():
    sim = Simulation(
        geo=geo_library['pei'](),
        ipm_builder=ipm_library['pei'](),
        mvm_builder=mm_library['pei']()
    )

    params = {
        'infection_duration': 4.0,
        'immunity_duration': 90.0,
        'infection_seed_loc': 0,
        'infection_seed_size': 10_000,
        'theta': 0.1,
        'move_control': 0.9,
    }

    sim.run(params, date(2015, 1, 1), 100)
    sim.run(params, date(2015, 1, 1), 100)
    sim.run(params, date(2015, 1, 1), 100)
    sim.run(params, date(2015, 1, 1), 100)
    sim.run(params, date(2015, 1, 1), 100)
    sim.run(params, date(2015, 1, 1), 100)
    sim.run(params, date(2015, 1, 1), 100)
    sim.run(params, date(2015, 1, 1), 100)


if __name__ == "__main__":
    main()
