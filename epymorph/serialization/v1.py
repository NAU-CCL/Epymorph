import importlib
import importlib.metadata
from collections import OrderedDict
from datetime import date
from functools import singledispatch
from typing import Annotated, Any, Literal, Self, TypeVar

import numpy as np
import sympy
from pydantic import BaseModel, Discriminator, Field

import epymorph.serialization.model as model
from epymorph.util import SaveParams


def serialize(obj: Any, **kwargs) -> Any:
    return to_serializable(obj, **kwargs)


def deserialize(obj: Any, **kwargs) -> Any:
    return from_serializable(obj, **kwargs)


@singledispatch
def to_serializable(obj: Any, **kwargs) -> Any:
    """Converts from a model type into its serializable companion type."""
    err = f"to_serializable for type '{obj.__class__.__name__}'"
    raise NotImplementedError(err)


@singledispatch
def from_serializable(obj: Any, **kwargs) -> Any:
    """Converts from a serializable type into its model companion type."""
    if isinstance(obj, BaseModel):
        err = f"from_serializable for type '{obj.__class__.__name__}'"
        raise NotImplementedError(err)
    return obj


T = TypeVar("T")


def _passthrough_handler(x: T, **kwargs) -> T:
    return x


def passthrough(data_type: type[T]):
    to_serializable.register(data_type)(_passthrough_handler)
    from_serializable.register(data_type)(_passthrough_handler)


passthrough(int)
passthrough(float)
passthrough(str)
passthrough(date)


@to_serializable.register
def _(obj: list, **kwargs) -> list:
    return [to_serializable(x, **kwargs) for x in obj]


@from_serializable.register
def _(obj: list, **kwargs) -> list:
    return [from_serializable(x, **kwargs) for x in obj]


@to_serializable.register
def _(obj: dict, **kwargs) -> dict:
    return {k: to_serializable(v, **kwargs) for k, v in obj.items()}


@from_serializable.register
def _(obj: dict, **kwargs) -> dict:
    return {k: from_serializable(v, **kwargs) for k, v in obj.items()}


SerializeT = TypeVar("SerializeT", bound=BaseModel)
ModelT = TypeVar("ModelT")


def serializes(model_type: type[ModelT]):
    """
    Class decorator that simplifies the definition of serialization pairs when
    they can be trivially converted to the other because they are constructed from
    the same set of attributes.
    """

    def serializes_decorator(serialize_type: type[SerializeT]):
        @to_serializable.register(model_type)
        def _(obj: ModelT, **kwargs) -> SerializeT:
            return serialize_type.model_validate(obj, from_attributes=True)

        @from_serializable.register(serialize_type)
        def _(obj: SerializeT, **kwargs) -> ModelT:
            return model_type(**from_serializable(dict(obj), **kwargs))

        return serialize_type

    return serializes_decorator


class NDArray(BaseModel):
    kind: Literal["NDArray"] = Field(default="NDArray", init=False)
    dtype: str
    values: list[str] | list[int] | list[float]


@to_serializable.register
def _(obj: np.ndarray, **kwargs) -> NDArray:
    if np.issubdtype(obj.dtype, np.str_):
        return NDArray(dtype="string", values=obj.tolist())
    if np.issubdtype(obj.dtype, np.integer):
        return NDArray(dtype="integer", values=obj.tolist())
    if np.issubdtype(obj.dtype, np.floating):
        return NDArray(dtype="floating", values=obj.tolist())
    err = f"Unhandled ndarray dtype: {str(obj.dtype)}"
    raise NotImplementedError(err)


@from_serializable.register
def _(obj: NDArray, **kwargs) -> np.ndarray:
    if obj.dtype == "string":
        return np.array(obj.values, dtype=np.str_)
    if obj.dtype == "integer":
        return np.array(obj.values, dtype=np.int64)
    if obj.dtype == "floating":
        return np.array(obj.values, dtype=np.float64)
    err = f"Unhandled ndarray dtype: {str(obj.dtype)}"
    raise NotImplementedError(err)


@serializes(model.CompartmentName)
class CompartmentName(BaseModel):
    base: str
    subscript: str | None = None
    strata: str | None = None


@serializes(model.CompartmentDef)
class CompartmentDef(BaseModel):
    name: CompartmentName
    tags: list[str]
    description: str | None


@serializes(model.EdgeName)
class EdgeName(BaseModel):
    compartment_from: CompartmentName
    compartment_to: CompartmentName


class EdgeDef(BaseModel):
    kind: Literal["EdgeDef"] = Field(default="EdgeDef", init=False)
    name: EdgeName
    rate: str


@to_serializable.register
def _(obj: model.EdgeDef, **kwargs) -> EdgeDef:
    return EdgeDef(
        name=to_serializable(obj.name, **kwargs),
        rate=str(obj.rate),
    )


@from_serializable.register
def _(
    obj: EdgeDef, *, symbols: dict[str, sympy.Symbol] | None = None, **kwargs
) -> model.EdgeDef:
    # TODO: WARNING -- this use of sympify is unsafe due to use of eval.
    # This is temporary until a better solution is found.
    # Also, should we have a conversion context in order to define a symbol
    # library ahead of time so that symbol references are global to the model?
    return model.EdgeDef(
        name=(name := from_serializable(obj.name, symbols=symbols, **kwargs)),
        rate=sympy.sympify(obj.rate, locals=symbols),
        compartment_from=sympy.Symbol(str(name.compartment_from)),
        compartment_to=sympy.Symbol(str(name.compartment_to)),
    )


class ForkDef(BaseModel):
    kind: Literal["ForkDef"] = Field(default="ForkDef", init=False)
    edges: list[EdgeDef]


@to_serializable.register
def _(obj: model.ForkDef, **kwargs) -> ForkDef:
    return ForkDef(edges=to_serializable(obj.edges, **kwargs))


@from_serializable.register
def _(
    obj: ForkDef, *, symbols: dict[str, sympy.Symbol] | None = None, **kwargs
) -> model.ForkDef:
    return model.fork(*from_serializable(obj.edges, symbols=symbols, **kwargs))


TransitionDef = Annotated[EdgeDef | ForkDef, Discriminator("kind")]

# Serialize as string:
# - DataShape
# - AttributeType
# - AbsoluteName
# - NamePattern


class AttributeDef(BaseModel):
    name: str
    type: str
    shape: str
    default_value: model.AttributeValue | None
    comment: str | None


@to_serializable.register
def _(obj: model.AttributeDef, **kwargs) -> AttributeDef:
    return AttributeDef(
        name=obj.name,
        type=model.dtype_serialize(obj.type),
        shape=str(obj.shape),
        default_value=obj.default_value,
        comment=obj.comment,
    )


@from_serializable.register
def _(obj: AttributeDef, **kwargs) -> model.AttributeDef:
    return model.AttributeDef(
        name=obj.name,
        type=model.dtype_deserialize(obj.type),
        shape=model.parse_shape(obj.shape),
        # TODO: we may need more careful parsing to make the default_value types align
        default_value=obj.default_value,
        comment=obj.comment,
    )


@serializes(model.TimeFrame)
class TimeFrame(BaseModel):
    start_date: date
    duration_days: int


class CustomScope(BaseModel):
    kind: Literal["CustomScope"] = Field(default="CustomScope", init=False)
    nodes: list[str]


@to_serializable.register
def _(obj: model.CustomScope, **kwargs) -> CustomScope:
    return CustomScope(nodes=obj.node_ids.tolist())


@from_serializable.register
def _(obj: CustomScope, **kwargs) -> model.CustomScope:
    return model.CustomScope(nodes=np.array(obj.nodes, dtype=np.str_))


def _serializes_census_scope(model_type: type[ModelT]):
    def serializes_decorator(serialize_type: type[SerializeT]):
        @to_serializable.register(model_type)
        def _(obj: ModelT, **kwargs) -> SerializeT:
            return serialize_type(
                year=obj.year,  # type: ignore
                includes_granularity=obj.includes_granularity,  # type: ignore
                includes=list(obj.includes),  # type: ignore
            )

        @from_serializable.register(serialize_type)
        def _(obj: SerializeT, **kwargs) -> ModelT:
            return model_type(
                year=obj.year,  # type: ignore
                includes_granularity=obj.includes_granularity,  # type: ignore
                includes=tuple(obj.includes),  # type: ignore
            )

        return serialize_type

    return serializes_decorator


@_serializes_census_scope(model.StateScope)
class StateScope(BaseModel):
    kind: Literal["StateScope"] = Field(default="StateScope", init=False)
    year: int
    includes_granularity: Literal["state"]
    includes: list[str]


@_serializes_census_scope(model.CountyScope)
class CountyScope(BaseModel):
    kind: Literal["CountyScope"] = Field(default="CountyScope", init=False)
    year: int
    includes_granularity: Literal["state", "county"]
    includes: list[str]


@_serializes_census_scope(model.TractScope)
class TractScope(BaseModel):
    kind: Literal["TractScope"] = Field(default="TractScope", init=False)
    year: int
    includes_granularity: Literal["state", "county", "tract"]
    includes: list[str]


@_serializes_census_scope(model.BlockGroupScope)
class BlockGroupScope(BaseModel):
    kind: Literal["BlockGroupScope"] = Field(default="BlockGroupScope", init=False)
    year: int
    includes_granularity: Literal["state", "county", "tract", "block group"]
    includes: list[str]


class DynamicClass(BaseModel):
    classpath: str
    args: list[Any]
    kwargs: dict[str, Any]

    @classmethod
    def _serialize(cls, obj: Any) -> Self:
        if isinstance(obj, SaveParams):
            args, kwargs = obj.init_params
            args = to_serializable(list(args))
            kwargs = to_serializable(kwargs)
        else:
            args = []
            kwargs = {}
        classpath = f"{obj.__class__.__module__}.{obj.__class__.__name__}"
        return cls(classpath=classpath, args=args, kwargs=kwargs)

    def _deserialize(self) -> Any:
        module, name = self.classpath.rsplit(".", 1)
        try:
            module = importlib.import_module(module)
            cls = getattr(module, name)
        except (ImportError, AttributeError) as e:
            err = f"Unable to deserialize object: '{self.classpath}'"
            raise ValueError(err) from e
        args = from_serializable(self.args)
        kwargs = from_serializable(self.kwargs)
        return cls(*args, **kwargs)


class SimulationFunction(DynamicClass):
    pass


@to_serializable.register
def _(obj: model.BaseSimulationFunction, **kwargs) -> SimulationFunction:
    return SimulationFunction._serialize(obj)


@from_serializable.register
def _(obj: SimulationFunction, **kwargs) -> Any:
    return obj._deserialize()


class GPM(BaseModel):
    name: str
    ipm: DynamicClass
    mm: DynamicClass
    init: SimulationFunction
    params: dict[str, Any] | None


@to_serializable.register
def _(obj: model.GPM, **kwargs) -> GPM:
    params = None
    if obj.params is not None:
        params = {str(k): to_serializable(v) for k, v in obj.params.items()}
    return GPM(
        name=obj.name,
        ipm=DynamicClass._serialize(obj.ipm),
        mm=DynamicClass._serialize(obj.mm),
        init=to_serializable(obj.init),
        params=params,
    )


@from_serializable.register
def _(obj: GPM, **kwargs) -> model.GPM:
    params = None
    if obj.params is not None:
        params = {
            model.ModuleNamePattern.parse(k): from_serializable(v)
            for k, v in obj.params.items()
        }
    return model.GPM(
        name=obj.name,
        ipm=obj.ipm._deserialize(),
        mm=obj.mm._deserialize(),
        init=from_serializable(obj.init),
        params=params,
    )


class SingleStrataRUME(BaseModel):
    kind: Literal["SingleStrataRUME"] = Field(default="SingleStrataRUME", init=False)
    strata: list[GPM]
    scope: Any
    time_frame: TimeFrame
    params: dict[str, Any]


@to_serializable.register
def _(obj: model.SingleStrataRUME, **kwargs) -> SingleStrataRUME:
    return SingleStrataRUME(
        strata=[to_serializable(x) for x in obj.strata],
        scope=to_serializable(obj.scope),
        time_frame=to_serializable(obj.time_frame),
        params={str(k): to_serializable(v) for k, v in obj.params.items()},
    )


@from_serializable.register
def _(obj: SingleStrataRUME, **kwargs) -> model.SingleStrataRUME:
    strata = [from_serializable(x) for x in obj.strata]
    if len(strata) > 1:
        err = "SingleStrataRUME contains more than one strata."
        raise ValueError(err)
    return model.SingleStrataRUME(
        strata=strata,
        ipm=strata[0].ipm,
        mms=OrderedDict((s.name, s.mm) for s in strata),
        scope=from_serializable(obj.scope),
        time_frame=from_serializable(obj.time_frame),
        params={
            model.NamePattern.parse(k): from_serializable(v)
            for k, v in obj.params.items()
        },
    )


class MultiStrataRUME(BaseModel):
    kind: Literal["MultiStrataRUME"] = Field(default="MultiStrataRUME", init=False)
    strata: list[GPM]
    meta_requirements: list[AttributeDef]
    meta_edges: list[TransitionDef]
    scope: Any
    time_frame: TimeFrame
    params: dict[str, Any]


@to_serializable.register
def _(obj: model.MultiStrataRUME, **kwargs) -> MultiStrataRUME:
    return MultiStrataRUME(
        strata=[to_serializable(x) for x in obj.strata],
        meta_requirements=to_serializable(obj.ipm.meta_requirements),
        meta_edges=to_serializable(obj.ipm.meta_edges),
        scope=to_serializable(obj.scope),
        time_frame=to_serializable(obj.time_frame),
        params={str(k): to_serializable(v) for k, v in obj.params.items()},
    )


@from_serializable.register
def _(obj: MultiStrataRUME, **kwargs) -> model.MultiStrataRUME:
    strata = [from_serializable(x) for x in obj.strata]
    return model.MultiStrataRUME(
        strata=strata,
        ipm=model.CombinedCompartmentModel(
            strata=[(gpm.name, gpm.ipm) for gpm in strata],
            meta_requirements=from_serializable(obj.meta_requirements),
            meta_edges=lambda _: from_serializable(obj.meta_edges),
        ),
        mms=model.remap_taus([(gpm.name, gpm.mm) for gpm in strata]),
        scope=from_serializable(obj.scope),
        time_frame=from_serializable(obj.time_frame),
        params={
            model.NamePattern.parse(k): from_serializable(v)
            for k, v in obj.params.items()
        },
    )


RUME = SingleStrataRUME | MultiStrataRUME


class SidecarFileWriter:
    files: dict[str, Any]

    def __init__(self):
        self.files = {}

    def write_npz(self, filename: str, source: object, data_attrs: list[str]) -> str:
        if filename in self.files:
            err = f"Unable to write npz file: file '{filename}' already exists"
            raise ValueError(err)

        data = {}
        for attr in data_attrs:
            if (value := getattr(source, attr, None)) is None:
                err = (
                    "Unable to write npz file: source object is missing attribute "
                    f"'{attr}'"
                )
                raise ValueError(err)
            if not isinstance(value, np.ndarray):
                err = (
                    f"Unable to write npz file: attribute '{attr}' is not a numpy array"
                )
                raise ValueError(err)
            data[attr] = value

        self.files[filename] = data
        return filename

    def save(self) -> None:
        pass


class SidecarFileReader:
    files: dict[str, Any]

    def __init__(self, files: dict[str, Any]):
        self.files = files

    @classmethod
    def load(cls) -> Self:
        return cls({})

    def read_npz(self, filename: str, data_attrs: list[str]) -> dict[str, np.ndarray]:
        if filename not in self.files:
            err = f"Unable to read npz file: file '{filename}' missing"
            raise ValueError(err)

        file_data = self.files[filename]
        if not isinstance(file_data, dict):
            err = f"Unable to read npz file: file '{filename}' is not npz format"
            raise ValueError(err)

        # TODO: type-check keys and values?

        data = {k: v for k, v in file_data.items() if k in data_attrs}
        if any((attr := x) not in data for x in data_attrs):
            err = (
                "Unable to read npz file: "
                f"file '{filename}' is missing attribute '{attr}'"
            )
            raise ValueError(err)

        return data


class Output(BaseModel):
    rume: RUME
    data_file: str


@to_serializable.register
def _(obj: model.Output, sidecar_files: SidecarFileWriter, **kwargs) -> Output:
    return Output(
        rume=to_serializable(obj.rume, sidecar_files=sidecar_files, **kwargs),
        data_file=sidecar_files.write_npz(
            "output.npz",
            obj,
            data_attrs=[
                "initial",
                "visit_compartments",
                "visit_events",
                "home_compartments",
                "home_events",
            ],
        ),
    )


@from_serializable.register
def _(obj: Output, sidecar_files: SidecarFileReader, **kwargs) -> model.Output:
    output_kwargs = sidecar_files.read_npz(
        "output.npz",
        data_attrs=[
            "initial",
            "visit_compartments",
            "visit_events",
            "home_compartments",
            "home_events",
        ],
    )
    return model.Output(
        rume=from_serializable(obj.rume, sidecar_files=sidecar_files, **kwargs),
        **output_kwargs,  # type: ignore
    )


###########################
# Serialization Envelopes #
###########################


class Envelope(BaseModel):
    serialization_version: Literal[1] = Field(default=1, init=False)
    epymorph_version: str = Field(
        default_factory=lambda: importlib.metadata.version("epymorph"),
        init=False,
    )


class SingleStrataRUMEEnvelope(Envelope):
    payload_type: Literal["SingleStrataRUME"]
    payload: SingleStrataRUME


class MultiStrataRUMEEnvelope(Envelope):
    payload_type: Literal["MultiStrataRUME"]
    payload: MultiStrataRUME


class OutputEnvelope(Envelope):
    payload_type: Literal["Output"]
    payload: Output
