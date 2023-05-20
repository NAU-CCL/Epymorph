"""
Implements the `run` subcommand executed from __main__.
"""
import re
import time
from datetime import date
from functools import wraps
from io import BufferedReader
from typing import Any, Callable, TypeVar

import matplotlib.pyplot as plt
import tomllib

from epymorph.data import geo_library, ipm_library, mm_library
from epymorph.simulation import Output, Simulation, configure_sim_logging
from epymorph.util import parse_duration, progress, stridesum

T = TypeVar('T')


def load_model(type: str, name: str, lib: dict[str, T]) -> T:
    """Load a model from the given library dictionary and print status checkbox."""
    text = f"{type} ({name})"
    print(f"[-] {text}", end="\r")
    result = lib.get(name)
    if result is not None:
        print(f"[✓] {text}")
        return result
    else:
        print(f"[X] {text}")
        raise Exception("ERROR: Unknown {type}: {name}")


def load_params(path: str) -> dict[str, Any]:
    """Load parameters from a file and print status checkbox."""
    text = f"Parameters (file:{path})"
    print(f"[-] {text}", end="\r")
    try:
        with open(path, 'rb') as file:
            result = tomllib.load(file)
        print(f"[✓] {text}")
        return result
    except Exception:
        print(f"[X] {text}")
        raise Exception(f"ERROR: Unable to load parameters: {path}")


def plot_event(out: Output, event_idx: int) -> None:
    """Charting: plot the event with the given index for all populations."""
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
    if out.ctx.nodes <= 12:
        ax.legend()
    fig.tight_layout()
    plt.show()


def plot_pop(out: Output, pop_idx: int) -> None:
    """Charting: plot all compartments (per 100k people) for the population at the given index."""
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
    if out.ctx.compartments <= 12:
        ax.legend()
    fig.tight_layout()
    plt.show()


# Exit codes:
# - 0 success
# - 1 invalid command
# - 2 error loading models/files
def run(ipm_name: str,
        mm_name: str,
        geo_name: str,
        start_date_str: str,
        duration_str: str,
        params_path: str,
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

    print("Loading requirements:")
    try:
        ipm_builder = load_model("IPM", ipm_name, ipm_library)
        mm_builder = load_model("MM", mm_name, mm_library)
        geo_builder = load_model("Geo", geo_name, geo_library)
        # TODO pull param defaults from models?
        params = load_params(params_path)
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

    # Draw a progress bar
    sim.on_start.subscribe(lambda _: print(progress(0.0), end='\r'))
    sim.on_tick.subscribe(lambda x: print(progress(x[1]), end='\r'))
    sim.on_end.subscribe(lambda _: print(progress(1.0)))

    t0 = time.perf_counter()
    out = sim.run(params, start_date, duration_days, progress=True)
    t1 = time.perf_counter()

    print(f"Runtime: {(t1 - t0):.3f}s")

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
