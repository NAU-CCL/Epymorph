"""
Implements the `prepare` subcommand executed from __main__.
"""
import os
from argparse import _SubParsersAction
from datetime import date

import numpy as np
import tomli_w
from numpy.typing import DTypeLike

from epymorph.cli.run import RunInput, interactive_select
from epymorph.data import geo_library, ipm_library, mm_library


def define_argparser(command_parser: _SubParsersAction):
    """
    Define `prepare` subcommand.
    ex: `epymorph prepare ./scratch/params.toml`
    """
    p = command_parser.add_parser(
        'prepare', help='prepare an input toml file for the run command')
    p.add_argument(
        'file',
        help='the path at which to save the file')
    p.add_argument(
        '--ipm',
        type=str,
        help='(optional) the name of an IPM from the library')
    p.add_argument(
        '--mm',
        type=str,
        help='(optional) the name of an MM from the library')
    p.add_argument(
        '--geo',
        type=str,
        help='(optional) the name of a Geo from the library')
    p.set_defaults(handler=lambda args: prepare_run_toml(
        out_path=args.file,
        ipm_name=args.ipm,
        mm_name=args.mm,
        geo_name=args.geo,
    ))


def _placeholder_value(dtype: DTypeLike):
    if np.issubdtype(dtype, np.int64):
        return 1
    elif np.issubdtype(dtype, np.float64):
        return 1.0
    elif np.issubdtype(dtype, np.str_):
        return "placeholder"


def prepare_run_toml(out_path: str,
                     ipm_name: str | None,
                     mm_name: str | None,
                     geo_name: str | None) -> int:
    """CLI command handler: create a skeleton toml input file."""

    # Exit codes:
    # - 0 success
    # - 1 unable to write file

    if os.path.exists(out_path):
        print(f"A file already exists at {out_path}")
        user_input = input("Overwrite file? (y/N): ").strip().lower()
        if user_input != 'y':
            print("Exited without altering existing file.")
            return 0  # success

    if ipm_name is None:
        ipm_name = interactive_select("IPM", ipm_library)

    if mm_name is None:
        mm_name = interactive_select("MM", mm_library)

    if geo_name is None:
        geo_name = interactive_select("GEO", geo_library)

    ipm = ipm_library[ipm_name]()
    mm = mm_library[mm_name]()

    attributes = {
        attrib.name: _placeholder_value(attrib.type)
        for attrib in ipm.attributes
        if attrib.source == 'params'
    } | {
        attrib.name: _placeholder_value(attrib.type)
        for attrib in mm.attributes
        if attrib.source == 'params'
    }

    document = RunInput(
        ipm=ipm_name,
        mm=mm_name,
        geo=geo_name,
        start_date=date.today(),
        duration_days=14,
        rng_seed=None,
        init={
            'initializer': 'single_location',
            'location': 0,
            'seed_size': 10_000,
        },
        params=attributes,
    )

    try:
        with open(out_path, mode="wb") as file:
            tomli_w.dump(document.model_dump(exclude_none=True), file)
        print(f"Wrote file at {out_path}")
        return 0  # success
    except Exception as e:
        print(e)
        return 1  # error writing toml file
