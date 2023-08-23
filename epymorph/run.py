"""
Implements the `run` subcommand executed from __main__.
"""
import re
import time
import tomllib
from datetime import date
from functools import partial
from typing import Any, TypeVar

import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, ValidationError

from epymorph.context import normalize_lists
from epymorph.data import geo_library, ipm_library, mm_library
from epymorph.initializer import initializer_library
from epymorph.movement.basic import BasicEngine
from epymorph.movement.engine import MovementEngine
from epymorph.movement.hypercube import HypercubeEngine
from epymorph.simulation import Output, Simulation, configure_sim_logging
from epymorph.util import Duration, progress, stridesum


def interactive_select(lib_name: str, lib: dict[str, Any]) -> str:
    keys = list(lib.keys())
    keys.sort()
    print(f"Select the {lib_name} you would like to use: ")
    for i, name in enumerate(keys):
        print(f'{i+1}. {name}')
    entry = input("Enter the number: ")
    print()
    index = int(entry) - 1
    return keys[index]


ModelT = TypeVar('ModelT')


def load_model(model_type: str, name: str, lib: dict[str, ModelT]) -> ModelT:
    """Load a model from the given library dictionary and print status checkbox."""
    text = f"{model_type} ({name})"
    print(f"[-] {text}", end="\r")
    result = lib.get(name)
    if result is not None:
        print(f"[✓] {text}")
        return result
    else:
        print(f"[X] {text}")
        raise Exception(f"ERROR: Unknown {model_type}: {name}")


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
            out.incidence[:, pop_idx, event_idx], len(out.ctx.clock.taus))
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


def save_npz(path: str, out: Output) -> None:
    """
    Save output prevalence and incidence as a compressed npz file.
    Key 'prevalence' will be a 3D array, of shape (T,P,C) -- just like it is in the Output object
    Key 'incidence' will be a 3D array, of shape (T,P,E) -- just like it is in the Output object
    """
    np.savez(path, prevalence=out.prevalence, incidence=out.incidence)
    # This can be loaded, for example as:
    # with load("./path/to/my-output-file.npz") as file:
    #     prev = file['prevalence']


def save_csv(path: str, out: Output) -> None:
    """
    Save output prevalence and incidence as a csv file.
    The data must be reshaped and labeled to fit a 2D format.
    Columns are: tick index, population index, then each compartment and then each event in IPM-specific order; ex:
    `t, p, c0, c1, c2, e0, e1, e2`
    """
    T, N, C, E = out.ctx.TNCE

    # reshape to 2d: (T,P,C) -> (T*P,C) and (T,P,E) -> (T*P,E)
    prv = np.reshape(out.prevalence, (T * N, C))
    inc = np.reshape(out.incidence, (T * N, E))

    # tick and pop index columns
    t_indices = np.reshape(np.repeat(np.arange(T), N), (T * N, 1))
    p_indices = np.reshape(np.tile(np.arange(N), T), (T * N, 1))

    data = np.concatenate((t_indices, p_indices, prv, inc), axis=1)
    c_labels = [f"c{i}" for i in range(C)]  # compartment headers
    e_labels = [f"e{i}" for i in range(E)]  # event headers
    header = "t,p," + ",".join(c_labels + e_labels)
    np.savetxt(path, data, fmt="%d", delimiter=",",
               header=header, comments="")


# Exit codes:
# - 0 success
# - 1 invalid input
# - 2 error loading models/files

class RunInput(BaseModel):
    """Pydantic model describing the contents of the input toml file."""
    ipm: str | None = None
    mm: str | None = None
    geo: str | None = None
    start_date: date
    duration: Duration
    rng_seed: int | None = None
    init: dict[str, Any]
    params: dict[str, Any]


def run(input_path: str,
        engine_id: str | None,
        out_path: str | None,
        chart: str | None,
        profiling: bool) -> int:
    """CLI command handler: run a simulation."""

    # Read input toml file.

    try:
        with open(input_path, "rb") as file:
            run_input = RunInput(**tomllib.load(file))
    except ValidationError as e:
        print(e)
        print(f"ERROR: missing required data in input file ({input_path})")
        return 1  # invalid input
    except OSError as e:
        print(e)
        print(f"ERROR: unable to open input file ({input_path})")
        return 1  # invalid input

    # Select engine.

    if engine_id is None:
        engine = None
    else:
        try:
            engine = dict[str, type[MovementEngine]]({
                'basic': BasicEngine,
                'hypercube': HypercubeEngine,
            })[engine_id]
        except KeyError:
            print(f"ERROR: Unknown engine: {engine_id}")
            return 2  # invalid input

    # Configure initializer.

    init_args = run_input.init
    try:
        init_fn = init_args.pop('initializer')
    except KeyError:
        print("ERROR: Required configuration `init.initializer` not found.")
        return 2  # invalid input
    try:
        init = initializer_library[init_fn]
    except KeyError:
        print(f"ERROR: Unknown initializer: {init_fn}")
        return 2  # invalid input
    initializer = partial(init, **normalize_lists(init_args))

    # Load models.

    try:
        ipm_name = run_input.ipm if run_input.ipm is not None \
            else interactive_select("IPM", ipm_library)

        mm_name = run_input.mm if run_input.mm is not None \
            else interactive_select("MM", mm_library)

        geo_name = run_input.geo if run_input.geo is not None \
            else interactive_select("GEO", geo_library)

        print("Loading requirements:")
        ipm_builder = load_model("IPM", ipm_name, ipm_library)
        mm_builder = load_model("MM", mm_name, mm_library)
        geo_builder = load_model("Geo", geo_name, geo_library)
    except Exception as e:
        print(e)
        return 2  # error loading models

    # TODO: model compatibility check
    # print("[✓] Model compatibility check")

    # Create and run simulation.

    start_date = run_input.start_date
    end_date = start_date + run_input.duration.to_relativedelta()
    duration_days = (end_date - start_date).days

    configure_sim_logging(enabled=not profiling)

    geo = geo_builder()
    sim = Simulation(geo, ipm_builder(), mm_builder(), engine)

    print()
    print(f"Running simulation ({sim.mvm_engine.__name__}):")
    print(f"• {start_date} to {end_date} ({duration_days} days)")
    print(f"• {geo.nodes} geo nodes")

    # Draw a progress bar
    sim.on_start.subscribe(lambda _: print(progress(0.0), end='\r'))
    sim.on_tick.subscribe(lambda x: print(progress(x[1]), end='\r'))
    sim.on_end.subscribe(lambda _: print(progress(1.0)))

    rng = None if run_input.rng_seed is None \
        else np.random.default_rng(run_input.rng_seed)

    t0 = time.perf_counter()
    out = sim.run(run_input.params, start_date, duration_days, initializer, rng)
    t1 = time.perf_counter()

    print(f"Runtime: {(t1 - t0):.3f}s")

    # Handle output.

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
                    print("Unable to display chart: there are not enough events!")
            elif chart_type == 'p':
                if chart_idx < out.ctx.nodes:
                    plot_pop(out, chart_idx)
                else:
                    print("Unable to display chart: there are not enough nodes!")

    if out_path is not None:
        if out_path.endswith(".npz"):
            print(f"Writing output to file: {out_path}")
            save_npz(out_path, out)
        elif out_path.endswith(".csv"):
            print(f"Writing output to file: {out_path}")
            save_csv(out_path, out)
        else:
            print(f"Unknown file format specified for output: {out_path}")

    print("Done")
    return 0  # exit code: success
