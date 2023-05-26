from typing import Callable

from epymorph.data.geo.pei import load as geo_pei_load
from epymorph.data.geo.sparsemod import load as geo_sparse_load
from epymorph.data.geo.us_counties_2015 import \
    load as geo_us_counties_2015_load
from epymorph.data.geo.us_states_2015 import load as geo_us_states_2015_load
from epymorph.data.ipm.no import load as ipm_no_load
from epymorph.data.ipm.pei import load as ipm_pei_load
from epymorph.data.ipm.simple_sirs import load as ipm_simple_sirs_load
from epymorph.movement import MovementBuilder, load_movement_spec
from epymorph.data.geo.single_pop_geo import load as single_pop_geo_load


def mm_loader(path) -> Callable[[], MovementBuilder]:
    def load() -> MovementBuilder:
        with open(path, 'r') as file:
            spec_string = file.read()
            return load_movement_spec(spec_string)
    return load


# THIS IS A PLACEHOLDER IMPLEMENTATION
# Ultimately we want to index the data directory at runtime.

ipm_library = {
    'no': ipm_no_load,
    'pei': ipm_pei_load,
    'simple_sirs': ipm_simple_sirs_load
}

mm_library = {
    'no': mm_loader('epymorph/data/mm/no.movement'),
    'icecube': mm_loader('epymorph/data/mm/icecube.movement'),
    'pei': mm_loader('epymorph/data/mm/pei.movement')
}

geo_library = {
    'pei': geo_pei_load,
    'us_counties_2015': geo_us_counties_2015_load,
    'us_states_2015': geo_us_states_2015_load,
    'sparse': geo_sparse_load,
    'single_pop_geo' : single_pop_geo_load
}
