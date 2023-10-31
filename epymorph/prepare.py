import os
from datetime import date

import tomli_w

from epymorph.data import geo_library, ipm_library, mm_library
from epymorph.run import RunInput, interactive_select


def prepare_run_toml(out_path: str,
                     ipm_name: str | None,
                     mm_name: str | None,
                     geo_name: str | None) -> int:
    """CLI command handler: create a skeleton toml input file."""

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
        params={
            # TODO: would like to inspect IPM/MM to figure
            # out which params are needed and list them here
            # But for now, just a placeholder...
            "example_parameter": 0.5,
        }
    )

    try:
        with open(out_path, mode="wb") as file:
            tomli_w.dump(document.model_dump(exclude_none=True), file)
        print(f"Wrote file at {out_path}")
        return 0  # success
    except Exception as e:
        print(e)
        return 1  # error writing toml file
