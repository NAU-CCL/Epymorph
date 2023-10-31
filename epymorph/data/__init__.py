"""
The library for epymorph's built-in IPMs, MMs, and GEOs.
"""
from importlib.abc import Traversable
from importlib.resources import as_file, files
from typing import Callable, TypeVar

from epymorph.compartment_model import CompartmentModel
from epymorph.data.geo.single_pop import load as load_geo_single_pop
from epymorph.data.ipm.no import load as load_ipm_no
from epymorph.data.ipm.pei import load as load_ipm_pei
from epymorph.data.ipm.seirs import load as load_ipm_seirs
from epymorph.data.ipm.sirh import load as load_ipm_sirh
from epymorph.data.ipm.sirs import load as load_ipm_sirs
from epymorph.data.ipm.sparsemod import load as load_ipm_sparsemod
from epymorph.geo.adrio import adrio_maker_library
from epymorph.geo.dynamic import DynamicGeo
from epymorph.geo.geo import Geo
from epymorph.geo.static import StaticGeo, StaticGeoFileOps
from epymorph.parser.movement import MovementSpec
from epymorph.util import as_sorted_dict


def mm_spec_loader(mm_spec_file: Traversable) -> Callable[[], MovementSpec]:
    """Returns a function to load the identified movement model."""
    def load() -> MovementSpec:
        with as_file(mm_spec_file) as file:
            spec_string = file.read_text(encoding="utf-8")
            return MovementSpec.load(spec_string)
    return load


def geo_spec_loader(geo_spec_file: Traversable) -> Callable[[], DynamicGeo]:
    """Returns a function to load the identified GEO (from spec)."""
    def load() -> DynamicGeo:
        with as_file(geo_spec_file) as file:
            return DynamicGeo.load(file, adrio_maker_library)
    return load


def geo_archive_loader(geo_archive_file: Traversable) -> Callable[[], StaticGeo]:
    """Returns a function to load a static geo from its archive file."""
    def load() -> StaticGeo:
        with as_file(geo_archive_file) as file:
            return StaticGeo.load(file)
    return load


# Model library registration.
# Models might be added to these libraries either directly or through
# an auto-discovery process


DATA_PATH = files('epymorph.data')
MM_PATH = DATA_PATH.joinpath('mm')
IPM_PATH = DATA_PATH.joinpath('ipm')
GEO_PATH = DATA_PATH.joinpath('geo')

ModelT = TypeVar('ModelT')

Library = dict[str, Callable[[], ModelT]]

ipm_library: Library[CompartmentModel] = as_sorted_dict({
    "no": load_ipm_no,
    "pei": load_ipm_pei,
    "sirs": load_ipm_sirs,
    "sirh": load_ipm_sirh,
    "seirs": load_ipm_seirs,
    "sparsemod": load_ipm_sparsemod
})
"""All epymorph intra-population models (by id)."""

mm_library: Library[MovementSpec] = as_sorted_dict({
    # Auto-discover all .movement files in the data/mm path.
    f.name.removesuffix('.movement'): mm_spec_loader(f)
    for f in MM_PATH.iterdir()
    if f.name.endswith('.movement')
})
"""All epymorph movement models (by id)."""

geo_library_static: Library[StaticGeo] = as_sorted_dict({
    # Auto-discover all .geo.tgz files in the data/geo path.
    name: geo_archive_loader(file)
    for file, name in StaticGeoFileOps.iterate_dir(GEO_PATH)
})
"""The subset of GEOs that are saved as archive files."""

geo_library_dynamic: Library[DynamicGeo] = as_sorted_dict({
    # Auto-discover all .geo (spec) files in the data/geo path.
    f.name.removesuffix('.geo'): geo_spec_loader(f)
    for f in GEO_PATH.iterdir()
    if f.name.endswith('.geo')
})
"""The subset of GEOs that are assembled through geospecs."""

geo_library: Library[Geo] = as_sorted_dict({
    # Combine static, dynamic, and Python geos.
    'single_pop': load_geo_single_pop,
    **geo_library_static,
    **geo_library_dynamic
})
"""All epymorph geo models (by id)."""
