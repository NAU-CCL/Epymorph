"""
Implements the `run` subcommand executed from __main__.
"""
import re
import time
from datetime import date
from typing import TypeVar

import matplotlib.pyplot as plt

from epymorph.data import geo_library, ipm_library, mm_library
from epymorph.simulation import Output, Simulation, configure_sim_logging
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


def plot_event(out: Output, event_idx: int) -> None:
    fig, ax = plt.subplots()
    ax.set_title(f"event {event_idx} incidence")
    ax.set_xlabel('days')
    ax.set_ylabel(f"e{event_idx}")
    x_axis = list(range(out.ctx.clock.num_days))
    for pop_idx in range(out.ctx.nodes):
        values = stridesum(
            out.incidence[:, pop_idx, event_idx], out.ctx.clock.num_steps)
        y_axis = values
        ax.plot(x_axis, y_axis, label=out.ctx.labels[pop_idx])
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_pop(out: Output, pop_idx: int) -> None:
    fig, ax = plt.subplots()
    ax.set_title(f"Prevalence in {out.ctx.labels[pop_idx]}")
    ax.set_xlabel('days')
    ax.set_ylabel('persons (log scale)')
    ax.set_yscale('log')
    # ax.set_ylim(bottom=1, top=10 ** 8)
    x_axis = [t.tausum for t in out.ctx.clock.ticks]
    compartments = [f"c{n}" for n in range(out.ctx.compartments)]
    for i, event in enumerate(compartments):
        y_axis = out.prevalence[:, pop_idx, i]
        ax.plot(x_axis, y_axis, label=event)
    ax.legend()
    fig.tight_layout()
    plt.show()



# Exit codes:
# - 0 success
# - 1 invalid command
# - 2 unknown model
def run(ipm_name: str,
        mm_name: str,
        geo_name: str,
        start_date_str: str,
        duration_str: str,
        chart: str | None,
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

    t0 = time.perf_counter()
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
    t1 = time.perf_counter()

    print(f"|#################################| 100% ({(t1 - t0):.3f}s)")
    
    # NOTE: this method of chart handling is a placeholder implementation
    if chart is not None:
        chart_regex = re.compile(r"^([ep])(\d+)$")
        match = chart_regex.match(chart)
        if match is None:
            print(f"Unknown chart type: {chart}")
        else:
            print(f"Displaying chart: {chart}")
            chart_type = match.group(1)
            chart_idx = int(match.group(2))

            if chart_type == 'e':
                if chart_idx < out.ctx.events:
                    plot_event(out, chart_idx)
                else:
                    print(f"Unable to display chart: there are not enough events!")
            elif chart_type == 'p':
                if chart_idx < out.ctx.nodes:
                    plot_pop(out, chart_idx)
                else:
                    print(f"Unable to display chart: there are not enough nodes!")

    print("Done")
    return 0  # exit code: success
