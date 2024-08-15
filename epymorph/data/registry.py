"""
Library creation and registration for built-in IPMs, MMs, and GEOs.
"""
from importlib import import_module
from importlib.abc import Traversable
from importlib.resources import as_file, files
from inspect import isclass, signature
from typing import Callable, Mapping, NamedTuple, TypeGuard, TypeVar

from epymorph.compartment_model import CompartmentModel
from epymorph.error import ModelRegistryException
from epymorph.geo.adrio import adrio_maker_library
from epymorph.geo.dynamic import DynamicGeo, DynamicGeoFileOps
from epymorph.geo.geo import Geo
from epymorph.geo.static import StaticGeo, StaticGeoFileOps
from epymorph.movement_model import MovementModel
from epymorph.util import as_sorted_dict

ModelT = TypeVar('ModelT')
LoaderFunc = Callable[[], ModelT]
Library = Mapping[str, LoaderFunc[ModelT]]
ClassLibrary = Mapping[str, type[ModelT]]


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

    def get_model_id(self, obj: object) -> str:
        """Retrieves the tagged model ID."""
        value = getattr(obj, self.annotation, None)
        if isinstance(value, str):
            return value
        else:
            raise ValueError("Unable to load model ID during model registry.")

    def set_model_id(self, obj: object, model_id: str) -> None:
        """Sets a model ID as a tag."""
        setattr(obj, self.annotation, model_id)


GEO_REGISTRY = _ModelRegistryInfo('geo')
IPM_REGISTRY = _ModelRegistryInfo('ipm')
MM_REGISTRY = _ModelRegistryInfo('mm')


# Model registration decorators


def ipm(ipm_id: str):
    """Decorates an IPM loader so we can register it with the system."""
    def make_decorator(model: type[CompartmentModel]) -> type[CompartmentModel]:
        IPM_REGISTRY.set_model_id(model, ipm_id)
        return model
    return make_decorator


def mm(mm_id: str):
    """Decorates an IPM loader so we can register it with the system."""
    def make_decorator(model: type[MovementModel]) -> type[MovementModel]:
        MM_REGISTRY.set_model_id(model, mm_id)
        return model
    return make_decorator


def geo(geo_id: str):
    """Decorates an IPM loader so we can register it with the system."""
    def make_decorator(func: LoaderFunc[Geo]) -> LoaderFunc[Geo]:
        GEO_REGISTRY.set_model_id(func, geo_id)
        return func
    return make_decorator


# Discovery and loading utilities


DiscoverT = TypeVar('DiscoverT', bound=CompartmentModel | MovementModel | StaticGeo)


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
        model.get_model_id(x): x
        for mod in modules
        for x in mod.__dict__.values()
        if callable(x) and x.__module__ == mod.__name__ and is_loader(x)
    }


def _discover_classes(model: _ModelRegistryInfo, library_type: type[DiscoverT]) -> ClassLibrary[DiscoverT]:
    """
    Search for the specified type of model, implemented by the specified Python class.
    """
    # There's nothing stopping you from calling this method with incompatible types,
    # you'll just probably come up empty in that scenario. But this is an internal method,
    # so no need to be over-careful.

    in_path = model.path
    modules = [
        import_module(f"{in_path}.{f.name.removesuffix('.py')}")
        for f in files(in_path).iterdir()
        if f.name != "__init__.py" and f.name.endswith('.py')
    ]

    return {
        model.get_model_id(x): x
        for mod in modules
        for x in mod.__dict__.values()
        if isclass(x) and issubclass(x, library_type) and x.__module__ == mod.__name__
    }


def _geo_spec_loader(geo_spec_file: Traversable) -> Callable[[], DynamicGeo]:
    """Returns a function to load the identified GEO (from spec)."""
    def load() -> DynamicGeo:
        with as_file(geo_spec_file) as file:
            return DynamicGeoFileOps.load_from_spec(file, adrio_maker_library)
    return load


def _geo_archive_loader(geo_archive_file: Traversable) -> Callable[[], StaticGeo]:
    """Returns a function to load a static geo from its archive file."""
    def load() -> StaticGeo:
        with as_file(geo_archive_file) as file:
            return StaticGeoFileOps.load_from_archive(file)
    return load


_GEO_DIR = files(GEO_REGISTRY.path)


# The model libraries (and useful library subsets)


ipm_library: ClassLibrary[CompartmentModel] = as_sorted_dict({
    **_discover_classes(IPM_REGISTRY, CompartmentModel),
})
"""All epymorph intra-population models (by id)."""


mm_library: ClassLibrary[MovementModel] = as_sorted_dict({
    **_discover_classes(MM_REGISTRY, MovementModel),
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
