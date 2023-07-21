from typing import Callable

from epymorph.data.geo.single_pop import load as geo_single_pop_load
from epymorph.data.ipm.no import load as ipm_no_load
from epymorph.data.ipm.pei import load as ipm_pei_load
from epymorph.data.ipm.sirh import load as ipm_sirh_load
from epymorph.data.ipm.sirs import load as ipm_sirs_load
from epymorph.data.ipm.sparsemod import load as ipm_sparsemod_load
from epymorph.geo import Geo, load_compressed_geo
from epymorph.movement import MovementBuilder, load_movement_spec


def mm_loader(id: str) -> Callable[[], MovementBuilder]:
    def load() -> MovementBuilder:
        with open(f"epymorph/data/mm/{id}.movement", "r") as file:
            spec_string = file.read()
            return load_movement_spec(spec_string)
    return load


def geo_npz_loader(id: str) -> Callable[[], Geo]:
    return lambda: load_compressed_geo(id)


# THIS IS A PLACEHOLDER IMPLEMENTATION
# Ultimately we want to index the data directory at runtime.

ipm_library = {
    "no": ipm_no_load,
    "pei": ipm_pei_load,
    "sirs": ipm_sirs_load,
    "sirh": ipm_sirh_load,
    "sparsemod": ipm_sparsemod_load
}

mm_library = {
    id: mm_loader(id)
    for id in ['no', 'icecube', 'pei', 'sparsemod']
}

geo_library = {
    'single_pop': geo_single_pop_load
} | {
    id: geo_npz_loader(id)
    for id in ['pei', 'us_counties_2015', 'us_states_2015', 'maricopa_cbg_2019']
}
