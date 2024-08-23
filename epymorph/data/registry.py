"""
Library registration for built-in IPMs and MMs.
"""
from importlib import import_module
from importlib.resources import files
from inspect import isclass
from typing import Callable, Mapping, NamedTuple, TypeVar

from epymorph.compartment_model import CompartmentModel
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


IPM_REGISTRY = _ModelRegistryInfo('ipm')
MM_REGISTRY = _ModelRegistryInfo('mm')


# Model registration decorators


def ipm(ipm_id: str):
    """Decorates an IPM class so we can register it with the system."""
    def make_decorator(model: type[CompartmentModel]) -> type[CompartmentModel]:
        IPM_REGISTRY.set_model_id(model, ipm_id)
        return model
    return make_decorator


def mm(mm_id: str):
    """Decorates an MM class so we can register it with the system."""
    def make_decorator(model: type[MovementModel]) -> type[MovementModel]:
        MM_REGISTRY.set_model_id(model, mm_id)
        return model
    return make_decorator


# Discovery and loading utilities


DiscoverT = TypeVar('DiscoverT', bound=CompartmentModel | MovementModel)


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


# The model libraries


ipm_library: ClassLibrary[CompartmentModel] = as_sorted_dict({
    **_discover_classes(IPM_REGISTRY, CompartmentModel),
})
"""All epymorph intra-population models (by id)."""


mm_library: ClassLibrary[MovementModel] = as_sorted_dict({
    **_discover_classes(MM_REGISTRY, MovementModel),
})
"""All epymorph movement models (by id)."""
