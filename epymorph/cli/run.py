"""
Implements the `run` subcommand executed from __main__.
"""

import re
import tomllib
from argparse import _SubParsersAction
from datetime import date
from functools import partial, wraps
from pathlib import Path
from typing import Any, Callable, Concatenate, ParamSpec, TypeVar

import numpy as np
from pydantic import BaseModel, ValidationError

import epymorph.plots as plots
from epymorph.compartment_model import CompartmentModel
from epymorph.data import Library, geo_library, ipm_library, mm_library
# TODO: CLI is broken
from epymorph.engine.output import Output
from epymorph.engine.standard_sim import StandardSimulation
from epymorph.error import UnknownModel
from epymorph.geo.adrio import adrio_maker_library
from epymorph.geo.cache import load_from_cache
from epymorph.geo.dynamic import DynamicGeoFileOps
from epymorph.geo.geo import Geo
from epymorph.geo.static import StaticGeoFileOps
from epymorph.initializer import initializer_library, normalize_init_params
from epymorph.log.messaging import sim_messaging
from epymorph.movement.parser import MovementSpec, parse_movement_spec
from epymorph.simulation import TimeFrame, default_rng, enable_logging


def define_argparser(command_parser: _SubParsersAction):
    """
    Define `run` subcommand.
    ex: `epymorph run ./scratch/params.toml --chart e0`
    """
    p = command_parser.add_parser(
        'run', help='run a simulation from library models')
    p.add_argument(
        'input',
        help='the path to an input toml file')
    p.add_argument(
        '-o', '--out',
        help="(optional) path to an output file to save the simulated prevalence data; specify either a .csv or .npz file")
    p.add_argument(
        '-c', '--chart',
        help='(optional) ID for chart to draw; "e0" for event incidence 0; "p2" for pop prevalence 2; etc. (this is a temporary feature in lieu of better output handling)')
    p.add_argument(
        '-p', '--profile',
        action='store_true',
        help='(optional) include this flag to run in profiling mode')
    p.add_argument(
        '-i', '--ignore_cache',
        action='store_true',
        help='(optional) include this flag to run the simulation without utilizing the Geo cache.')
    p.add_argument(
        '-m', '--mute_geo',
        action='store_false',
        help='(optional) include this flag to silence geo data retreival messaging.')

    p.set_defaults(handler=lambda args: run(
        input_path=args.input,
        out_path=args.out,
        chart=args.chart,
        profiling=args.profile,
        ignore_cache=args.ignore_cache,
        geo_messaging=args.mute_geo
    ))


class RunInput(BaseModel):
    """Pydantic model describing the contents of the input toml file."""
    ipm: str | None = None
    mm: str | None = None
    geo: str | None = None
    start_date: date
    duration_days: int
    rng_seed: int | None = None
    init: dict[str, Any]
    params: dict[str, Any]


def run(input_path: str,
        out_path: str | None,
        chart: str | None,
        profiling: bool,
        ignore_cache: bool,
        geo_messaging: bool) -> int:
    """CLI command handler: run a simulation."""

    # Exit codes:
    # - 0 success
    # - 1 invalid input
    # - 2 error loading models/files

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

    # Configure initializer.

    init_params = run_input.init.copy()
    try:
        init_fn = init_params.pop('initializer')
    except KeyError:
        print("ERROR: Required configuration `init.initializer` not found.")
        return 2  # invalid input
    try:
        init = initializer_library[init_fn]
    except KeyError:
        print(f"ERROR: Unknown initializer: {init_fn}")
        return 2  # invalid input

    initializer = partial(init, **normalize_init_params(init_params))

    # Load models.

    try:
        ipm_name = run_input.ipm if run_input.ipm is not None \
            else interactive_select("IPM", ipm_library)

        mm_name = run_input.mm if run_input.mm is not None \
            else interactive_select("MM", mm_library)

        geo_name = run_input.geo if run_input.geo is not None \
            else interactive_select("GEO", geo_library)

        print("Loading requirements:")
        ipm = load_model_ipm(ipm_name)
        mm = load_model_mm(mm_name)
        geo = load_model_geo(geo_name, ignore_cache)
    except Exception as e:
        print(e)
        return 2  # error loading models

    # Create and run simulation.

    sim = StandardSimulation(
        geo, ipm, mm,
        run_input.params,
        TimeFrame(run_input.start_date, run_input.duration_days),
        initializer,
        default_rng(run_input.rng_seed)
    )

    if not profiling:
        enable_logging()

    # Run simulation with appropriate messaging contexts

    with sim_messaging(sim, geo_messaging):
        out = sim.run()

    # Draw charts (if specified).
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
                if chart_idx < out.dim.events:
                    plots.plot_event(out, chart_idx)
                else:
                    print("Unable to display chart: there are not enough events!")
            elif chart_type == 'p':
                if chart_idx < out.dim.nodes:
                    plots.plot_pop(out, chart_idx)
                else:
                    print("Unable to display chart: there are not enough nodes!")

    # Write output to file (if specified).
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


def interactive_select(model_type: str, lib: Library) -> str:
    """
    Provide an interactive selection for a model, allowing the user
    to pick an implementation from the built-in model library.
    """
    keys = list(lib.keys())
    keys.sort()
    print(f"Select the {model_type} you would like to use: ")
    for i, name in enumerate(keys):
        print(f'{i+1}. {name}')
    entry = input("Enter the number: ")
    print()
    index = int(entry) - 1
    return keys[index]


ModelT = TypeVar('ModelT')
P = ParamSpec('P')


def load_messaging(description: str):
    """
    Decorates a loading function to make it emit pretty messages.
    Requires that the first parameter is a name.
    """
    def make_decorator(func: Callable[Concatenate[str, P], ModelT]) -> Callable[Concatenate[str, P], ModelT]:
        @wraps(func)
        def decorator(name: str, *args: P.args, **kwargs: P.kwargs) -> ModelT:
            full_description = f"{description} ({name})"
            try:
                print(f"[-] {full_description}", end="\r")
                value = func(name, *args, **kwargs)
                print(f"[✓] {full_description}")
                return value
            except Exception:
                print(f"[X] {full_description}")
                raise
        return decorator
    return make_decorator


@load_messaging("GEO")
def load_model_geo(name_or_path: str, ignore_cache: bool) -> Geo:
    """Loads a geo by name."""
    if not ignore_cache:
        cached = load_from_cache(name_or_path)
        if cached is not None:
            return cached

    if name_or_path in geo_library:
        return geo_library[name_or_path]()

    path = Path(name_or_path).expanduser()
    if not path.exists():
        raise UnknownModel('GEO', name_or_path)

    # check for non-library geo cached
    if not ignore_cache:
        cached = load_from_cache(path.stem)
        if cached is not None:
            return cached

    if path.suffix == ".geo":
        return DynamicGeoFileOps.load_from_spec(path, adrio_maker_library)
    elif path.suffix == ".tar":
        return StaticGeoFileOps.load_from_archive(path)
    else:
        msg = "Invalid file type. A geo can only be loaded from a .geo or .geo.tar file."
        raise Exception(msg)


@load_messaging("IPM")
def load_model_ipm(name: str) -> CompartmentModel:
    """Loads an IPM by name."""
    if name not in ipm_library:
        raise UnknownModel('IPM', name)

    return ipm_library[name]()


@load_messaging("MM")
def load_model_mm(name_or_path: str) -> MovementSpec:
    """Loads a movement model by name or path."""
    if name_or_path in mm_library:
        return mm_library[name_or_path]()

    path = Path(name_or_path)
    if not path.exists():
        raise UnknownModel('MM', name_or_path)

    with open(path, mode='r', encoding='utf-8') as file:
        spec_string = file.read()
        return parse_movement_spec(spec_string)


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
    T, N, C, E = out.dim.TNCE

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
