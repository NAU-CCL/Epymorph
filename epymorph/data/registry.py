"""
Library creation and registration for built-in IPMs, MMs, and GEOs.
"""
from importlib import import_module
from importlib.abc import Traversable
from importlib.resources import as_file, files
from inspect import signature
from typing import Callable, NamedTuple, TypeGuard, TypeVar, cast

from epymorph.compartment_model import CompartmentModel
from epymorph.error import ModelRegistryException
from epymorph.geo.adrio import adrio_maker_library
from epymorph.geo.dynamic import DynamicGeo
from epymorph.geo.geo import Geo
from epymorph.geo.static import StaticGeo, StaticGeoFileOps
from epymorph.parser.movement import MovementSpec
from epymorph.util import as_sorted_dict

ModelT = TypeVar('ModelT')
LoaderFunc = Callable[[], ModelT]
Library = dict[str, LoaderFunc[ModelT]]


class _ModelRegistryInfo(NamedTuple):
    """Info about how a model type should be managed by epymorph's model library."""
    name: str

    @property
    def path(self) -> str:
        """Get the data path for this model type."""
        return f'epymorph.data.{self.name}'

    @property
    def annotation(self) -> str:
        """Get the ID annotation name for this model type."""
        return f'__epymorph_{self.name}_id__'

    def get_model_id(self, func: Callable) -> str | None:
        """Retrieves the tagged model ID."""
        value = getattr(func, self.annotation, None)
        return value if isinstance(value, str) else None

    def set_model_id(self, func: Callable, model_id: str) -> None:
        """Sets a model ID as a tag."""
        setattr(func, self.annotation, model_id)


GEO_REGISTRY = _ModelRegistryInfo('geo')
IPM_REGISTRY = _ModelRegistryInfo('ipm')
MM_REGISTRY = _ModelRegistryInfo('mm')


# Model registration decorators


def ipm(ipm_id: str):
    """Decorates an IPM loader so we can register it with the system."""
    def make_decorator(func: LoaderFunc[CompartmentModel]) -> LoaderFunc[CompartmentModel]:
        IPM_REGISTRY.set_model_id(func, ipm_id)
        return func
    return make_decorator


def mm(mm_id: str):
    """Decorates an IPM loader so we can register it with the system."""
    def make_decorator(func: LoaderFunc[MovementSpec]) -> LoaderFunc[MovementSpec]:
        MM_REGISTRY.set_model_id(func, mm_id)
        return func
    return make_decorator


def geo(geo_id: str):
    """Decorates an IPM loader so we can register it with the system."""
    def make_decorator(func: LoaderFunc[Geo]) -> LoaderFunc[Geo]:
        GEO_REGISTRY.set_model_id(func, geo_id)
        return func
    return make_decorator


# Discovery and loading utilities


DiscoverT = TypeVar('DiscoverT', bound=CompartmentModel | MovementSpec | StaticGeo)


def _discover(model: _ModelRegistryInfo, library_type: type[DiscoverT]) -> Library[DiscoverT]:
    """
    Search for the specified type of model, implemented by the specified Python class.
    """
    # There's nothing stopping you from calling this method with incompatible types,
    # you'll just probably come up empty in that scenario. But this is an internal method,
    # so no need to be over-careful.

    def is_loader(func: Callable) -> TypeGuard[LoaderFunc[DiscoverT]]:
        """Is `func` an acceptable model loader?"""
        if not callable(func):
            return False
        if model.get_model_id(func) is None:
            return False
        sig = signature(func, eval_str=True)
        if len(sig.parameters) > 0 or sig.return_annotation != library_type:
            msg = f"""\
Attempted to register model of type '{model.name}' with an invalid method signature.
See function '{func.__name__}' in {func.__module__}
The function must take zero parameters and its return-type must be correctly annotated ({library_type.__name__})."""
            raise ModelRegistryException(msg)
        return True

    in_path = model.path
    modules = [
        import_module(f"{in_path}.{f.name.removesuffix('.py')}")
        for f in files(in_path).iterdir()
        if f.name != "__init__.py" and f.name.endswith('.py')
    ]

    return {
        cast(str, model.get_model_id(x)): x
        for mod in modules
        for x in mod.__dict__.values()
        if callable(x) and x.__module__ == mod.__name__ and is_loader(x)
    }


def _mm_spec_loader(mm_spec_file: Traversable) -> Callable[[], MovementSpec]:
    """Returns a function to load the identified movement model."""
    def load() -> MovementSpec:
        with as_file(mm_spec_file) as file:
            spec_string = file.read_text(encoding="utf-8")
            return MovementSpec.load(spec_string)
    return load


def _geo_spec_loader(geo_spec_file: Traversable) -> Callable[[], DynamicGeo]:
    """Returns a function to load the identified GEO (from spec)."""
    def load() -> DynamicGeo:
        with as_file(geo_spec_file) as file:
            return DynamicGeo.load(file, adrio_maker_library)
    return load


def _geo_archive_loader(geo_archive_file: Traversable) -> Callable[[], StaticGeo]:
    """Returns a function to load a static geo from its archive file."""
    def load() -> StaticGeo:
        with as_file(geo_archive_file) as file:
            return StaticGeo.load(file)
    return load


_MM_DIR = files(MM_REGISTRY.path)
_GEO_DIR = files(GEO_REGISTRY.path)


# The model libraries (and useful library subsets)


ipm_library: Library[CompartmentModel] = as_sorted_dict({
    **_discover(IPM_REGISTRY, CompartmentModel),
})
"""All epymorph intra-population models (by id)."""

mm_library_parsed: Library[MovementSpec] = as_sorted_dict({
    # Auto-discover all .movement files in the data/mm path.
    f.name.removesuffix('.movement'): _mm_spec_loader(f)
    for f in _MM_DIR.iterdir()
    if f.name.endswith('.movement')
})
"""The subset of MMs that are parsed from movement files."""

mm_library: Library[MovementSpec] = as_sorted_dict({
    **_discover(MM_REGISTRY, MovementSpec),
    **mm_library_parsed,
})
"""All epymorph movement models (by id)."""

geo_library_static: Library[StaticGeo] = as_sorted_dict({
    # Auto-discover all .geo.tgz files in the data/geo path.
    name: _geo_archive_loader(file)
    for file, name in StaticGeoFileOps.iterate_dir(_GEO_DIR)
})
"""The subset of GEOs that are saved as archive files."""

geo_library_dynamic: Library[DynamicGeo] = as_sorted_dict({
    # Auto-discover all .geo (spec) files in the data/geo path.
    f.name.removesuffix('.geo'): _geo_spec_loader(f)
    for f in _GEO_DIR.iterdir()
    if f.name.endswith('.geo')
})
"""The subset of GEOs that are assembled through geospecs."""

geo_library: Library[Geo] = as_sorted_dict({
    # Combine static, dynamic, and Python geos.
    **_discover(GEO_REGISTRY, StaticGeo),
    **geo_library_static,
    **geo_library_dynamic,
})
"""All epymorph geo models (by id)."""
