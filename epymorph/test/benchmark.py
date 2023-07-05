from datetime import date
from statistics import quantiles, stdev
from time import perf_counter

import numpy as np

from epymorph.data import geo_library, ipm_library, mm_library
from epymorph.simulation import Simulation


def setup_test():
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

    return lambda: sim.run(params, date(2015, 1, 1), 100)


def main():
    test = setup_test()
    test()  # warmup

    n = 4
    times = np.zeros(n * 4, dtype=float)

    print("Running benchmark", end="", flush=True)
    for i in range(n):
        j = i * 4
        t0 = perf_counter()
        test()
        t1 = perf_counter()
        times[j + 0] = 1000.0 * (t1 - t0)
        t0 = perf_counter()
        test()
        t1 = perf_counter()
        times[j + 1] = 1000.0 * (t1 - t0)
        t0 = perf_counter()
        test()
        t1 = perf_counter()
        times[j + 2] = 1000.0 * (t1 - t0)
        t0 = perf_counter()
        test()
        t1 = perf_counter()
        times[j + 3] = 1000.0 * (t1 - t0)
        print(".", end="", flush=True)
    print("")

    qs = quantiles(times)

    print(f"           min: {min(times):9.3f} ms")
    print(f"lower quartile: {qs[0]:9.3f} ms")
    print(f"          mean: {qs[1]:9.3f} ms")
    print(f"upper quartile: {qs[2]:9.3f} ms")
    print(f"           max: {max(times):9.3f} ms")
    print(f"     std. dev.: {stdev(times):9.3f} ms")


if __name__ == "__main__":
    main()
