from importlib.abc import Traversable
from importlib.resources import files
from typing import Callable

from epymorph.data.geo.maricopa_cbg_2019 import load as maricopa_cbg_2019_load
from epymorph.data.geo.pei import load as geo_pei_load
from epymorph.data.geo.single_pop_geo import load as single_pop_geo_load
from epymorph.data.geo.sparsemod import load as geo_sparsemod_load
from epymorph.data.geo.us_counties_2015 import \
    load as geo_us_counties_2015_load
from epymorph.data.geo.us_states_2015 import load as geo_us_states_2015_load
from epymorph.data.ipm.basis_sdh import load as ipm_basis_sdh_load
from epymorph.data.ipm.no import load as ipm_no_load
from epymorph.data.ipm.pei import load as ipm_pei_load
from epymorph.data.ipm.sdh import load as ipm_sdh_load
from epymorph.data.ipm.simple_sirs import load as ipm_simple_sirs_load
from epymorph.data.ipm.sirh import load as ipm_sirh_load
from epymorph.movement import MovementBuilder, load_movement_spec

DATA_PATH = files('epymorph.data')
MM_PATH = DATA_PATH.joinpath('mm')
IPM_PATH = DATA_PATH.joinpath('ipm')
GEO_PATH = DATA_PATH.joinpath('geo')


def mm_loader(mm_file: Traversable) -> Callable[[], MovementBuilder]:
    def load() -> MovementBuilder:
        spec_string = mm_file.read_text(encoding='utf-8')
        return load_movement_spec(spec_string)
    return load


# THIS IS A PLACEHOLDER IMPLEMENTATION
# Ultimately we want to index the data directory at runtime.

ipm_library = {
    "no": ipm_no_load,
    "pei": ipm_pei_load,
    "simple_sirs": ipm_simple_sirs_load,
    "sirh": ipm_sirh_load,
    "sdh": ipm_sdh_load,
    "basis_sdh": ipm_basis_sdh_load,
}


mm_library = {
    f.name.removesuffix('.movement'): mm_loader(f)
    for f in MM_PATH.iterdir()
    if f.name.endswith('.movement')
}


geo_library = {
    "pei": geo_pei_load,
    "us_counties_2015": geo_us_counties_2015_load,
    "us_states_2015": geo_us_states_2015_load,
    "maricopa_cbg_2019": maricopa_cbg_2019_load,
    "sparsemod": geo_sparsemod_load,
    "single_pop_geo": single_pop_geo_load,
}
