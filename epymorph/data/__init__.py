"""
The library for epymorph's built-in IPMs, MMs, and GEOs.
"""
from importlib.abc import Traversable
from importlib.resources import as_file, files
from typing import Callable, TypeVar, cast

from epymorph.data.geo.single_pop import load as geo_single_pop_load
from epymorph.data.ipm.no import load as ipm_no_load
from epymorph.data.ipm.pei import load as ipm_pei_load
from epymorph.data.ipm.seirs import load as ipm_seirs_load
from epymorph.data.ipm.sirh import load as ipm_sirh_load
from epymorph.data.ipm.sirs import load as ipm_sirs_load
from epymorph.data.ipm.sparsemod import load as ipm_sparsemod_load
from epymorph.geo import Geo, GEOBuilder, load_compressed_geo
from epymorph.ipm.ipm import IpmBuilder
from epymorph.movement.dynamic import DynamicMovementBuilder
from epymorph.movement.engine import MovementBuilder
from epymorph.parser.movement import MovementSpec, movement_spec
from epymorph.util import as_sorted_dict

DATA_PATH = files('epymorph.data')
MM_PATH = DATA_PATH.joinpath('mm')
IPM_PATH = DATA_PATH.joinpath('ipm')
GEO_PATH = DATA_PATH.joinpath('geo')


def mm_loader(mm_file: Traversable) -> Callable[..., MovementBuilder]:
    """Returns a function to load the identified movement model."""
    def load() -> MovementBuilder:
        spec_string = mm_file.read_text(encoding="utf-8")
        results = movement_spec.parse_string(spec_string, parse_all=True)
        spec = cast(MovementSpec, results[0])
        return DynamicMovementBuilder(spec)
    return load


def geo_spec_loader(geo_spec_file: Traversable) -> Callable[..., Geo]:
    """Returns a function to load the identified GEO (from spec)."""
    def load(force=False) -> Geo:
        spec_string = geo_spec_file.read_text(encoding="utf-8")
        return GEOBuilder.from_spec(spec_string).build(force)
    return load


def geo_npz_loader(geo_npz_file: Traversable) -> Callable[..., Geo]:
    """Returns a function to load the identified GEO (from npz)."""
    def load() -> Geo:
        with as_file(geo_npz_file) as file:
            return load_compressed_geo(file)
    return load


# THIS IS A PLACEHOLDER IMPLEMENTATION
# Ultimately we want to index the data directory at runtime.

ModelT = TypeVar('ModelT')

Library = dict[str, Callable[..., ModelT]]


ipm_library: Library[IpmBuilder] = as_sorted_dict({
    "no": ipm_no_load,
    "pei": ipm_pei_load,
    "sirs": ipm_sirs_load,
    "sirh": ipm_sirh_load,
    "seirs": ipm_seirs_load,
    "sparsemod": ipm_sparsemod_load
})
"""All epymorph intra-population models (by id)."""


mm_library: Library[MovementBuilder] = as_sorted_dict({
    f.name.removesuffix('.movement'): mm_loader(f)
    for f in MM_PATH.iterdir()
    if f.name.endswith('.movement')
})
"""All epymorph movement models (by id)."""


geo_library_compressed: Library[Geo] = as_sorted_dict({
    f.name.removesuffix('_geo.npz'): geo_npz_loader(f)
    for f in GEO_PATH.iterdir()
    if f.name.endswith('_geo.npz')
})
"""The subset of GEOs that are saved as npz files."""

geo_library_cachable: Library[Geo] = as_sorted_dict({
    f.name.removesuffix('.geo'): geo_spec_loader(f)
    for f in GEO_PATH.iterdir()
    if f.name.endswith('.geo')
})
"""The subset of GEOs that use the GEOBuilder interface."""

geo_library: Library[Geo] = as_sorted_dict({
    'single_pop': geo_single_pop_load,
    **geo_library_compressed,
    **geo_library_cachable
})
"""All epymorph geo models (by id)."""
