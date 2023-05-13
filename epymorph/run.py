"""
Implements the `run` subcommand executed from __main__.
"""
from datetime import date
from typing import TypeVar

import matplotlib.pyplot as plt

from epymorph.data import geo_library, ipm_library, mm_library
from epymorph.simulation import Simulation, configure_sim_logging
from epymorph.util import parse_duration, stridesum

T = TypeVar('T')

def _check_model(type: str, name: str, lib: dict[str, T]) -> T:
    print(f"[-] {type} ({name})", end="\r")
    obj = lib.get(name)
    if obj is None:
        print(f"[X] {type} ({name})")
        raise Exception(f"ERROR: Unknown {type}: {name}")
    else:
        print(f"[✓] {type} ({name})\r")
        return obj

# Exit codes:
# - 0 success
# - 1 invalid command
# - 2 unknown model
def run(ipm_name: str,
        mm_name: str,
        geo_name: str,
        start_date_str: str,
        duration_str: str,
        profiling: bool) -> int:
    """Run a simulation. Returns exit code."""
    
    duration = parse_duration(duration_str)
    if duration is None:
        print(f"ERROR: invalid duration ({duration_str})")
        return 1
    start_date = date.fromisoformat(start_date_str)
    end_date = start_date + duration
    duration_days = (end_date - start_date).days

    print("Checking models:")

    try:
        ipm_builder = _check_model("IPM", ipm_name, ipm_library)
        mm_builder = _check_model("MM", mm_name, mm_library)
        geo_builder = _check_model("Geo", geo_name, geo_library)
    except Exception as e:
        print(e)
        return 2

    # TODO: model compatibility check
    # print("[✓] Model compatibility check")
    
    configure_sim_logging(enabled=not profiling)

    geo = geo_builder()
    sim = Simulation(geo, ipm_builder(), mm_builder())

    print()
    print(f"Running simulation:")
    print(f"• {start_date} to {end_date} ({duration_days} days)")
    print(f"• {geo.nodes} geo nodes")
    print("|                                 | 0%  ", end="\r")

    # TODO: how to handle params?

    out = sim.run(
        param={
            'theta': 0.1,
            'move_control': 0.9,
            'infection_duration': 4.0,
            'immunity_duration': 90.0,
            'infection_seed_loc': 0,
            'infection_seed_size': 10_000
        },
        start_date=start_date,
        duration_days=duration_days
    )

    print("|#################################| 100%")
    print("Displaying charts...")
    
    event = 0
    fig, ax = plt.subplots()
    ax.set_title('Infection incidence')
    ax.set_xlabel('days')
    ax.set_ylabel('infections')
    x_axis = list(range(out.ctx.clock.num_days))
    for pop_idx in range(geo.nodes):
        values = stridesum(
            out.incidence[:, pop_idx, event], out.ctx.clock.num_steps)
        y_axis = values
        ax.plot(x_axis, y_axis, label=geo.labels[pop_idx])
    ax.legend()
    fig.tight_layout()
    plt.show()

    print("Done")
    return 0  # exit code: success
