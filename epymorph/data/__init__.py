from typing import Callable

from epymorph.data.geo.pei import load as geo_pei_load
from epymorph.data.ipm.no import load as ipm_no_load
from epymorph.data.ipm.pei import load as ipm_pei_load
from epymorph.movement import MovementBuilder, load_movement_spec


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
    'pei': ipm_pei_load
}

mm_library = {
    'no': mm_loader('epymorph/data/mm/no.movement'),
    'icecube': mm_loader('epymorph/data/mm/icecube.movement'),
    'pei': mm_loader('epymorph/data/mm/pei.movement')
}

geo_library = {
    'pei': geo_pei_load
}
