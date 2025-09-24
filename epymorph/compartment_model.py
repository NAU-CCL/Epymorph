"""
The basis of the intra-population model (disease mechanics, aka IPM) system in epymorph.
This represents disease mechanics using a compartmental model for tracking populations
as groupings of integer-numbered individuals.
"""

import dataclasses
import re
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Literal,
    NamedTuple,
    OrderedDict,
    Self,
    Sequence,
    Type,
    TypeVar,
    final,
)
from warnings import warn

import numpy as np
from numpy.typing import NDArray
from sympy import Add, Expr, Float, Integer, Symbol
from typing_extensions import override

from epymorph.attribute import AbsoluteName, AttributeDef
from epymorph.error import IPMValidationError
from epymorph.strata import DEFAULT_STRATA, META_STRATA, gpm_strata
from epymorph.sympy_shim import simplify, simplify_sum, substitute, to_symbol
from epymorph.tools.ipm_diagram import render_diagram
from epymorph.util import (
    acceptable_name,
    are_instances,
    are_unique,
    filter_unique,
)

######################
# Model Compartments #
######################

BIRTH = Symbol("birth_exogenous")
"""
An IPM psuedo-compartment representing exogenous input of individuals.
This is useful in defining IPM edges.
"""

DEATH = Symbol("death_exogenous")
"""
An IPM psuedo-compartment representing exogenous removal of individuals.
This is useful in defining IPM edges.
"""

exogenous_states = (BIRTH, DEATH)
"""The list of supported exogenous states."""


@dataclass(frozen=True)
class CompartmentName:
    """
    The name of a compartment, which may have subscript and strata parts.

    Parameters
    ----------
    base :
        The base name of the compartment.
    subscript :
        The optional subscript part of the name.
    strata :
        The optional strata part of the name.
    """

    base: str
    """The base name of the compartment."""
    subscript: str | None
    """The optional subscript part of the name."""
    strata: str | None
    """The optional strata part of the name."""
    full: str = field(init=False, hash=False, compare=False)
    """The full name as a string, with all parts combined with underscores."""

    def __post_init__(self):
        full = "_".join(
            x for x in (self.base, self.subscript, self.strata) if x is not None
        )
        if acceptable_name.match(full) is None:
            raise ValueError(f"Invalid compartment name: {full}")
        object.__setattr__(self, "full", full)

    def with_subscript(self, subscript: str | None) -> Self:
        """
        Return a copy of this name with the subscript changed.

        Parameters
        ----------
        subscript :
            The new subscript.

        Returns
        -------
        :
            The new name.
        """
        if self.subscript == "exogenous":
            return self
        return dataclasses.replace(self, subscript=subscript)

    def with_strata(self, strata: str | None) -> Self:
        """
        Return a copy of this name with the strata changed.

        Parameters
        ----------
        strata :
            The new strata.

        Returns
        -------
        :
            The new name.
        """
        if self.subscript == "exogenous":
            return self
        return dataclasses.replace(self, strata=strata)

    def __str__(self) -> str:
        return self.full

    @classmethod
    def parse(cls, name: str) -> Self:
        """
        Parse a string as a `CompartmentName`. If the name contains no underscores,
        the entire name is the base name. If the name contains at least one underscore,
        the part before the first underscore is the base name and everything after is
        the subscript part. It is not possible to create a stratified name this way.

        For example: in "E_phase_1", "E" is the base name and "phase_1" is the
        subscript.

        Parameters
        ----------
        name :
            The string to parse.

        Returns
        -------
        :
            The parsed compartment name.
        """
        if (i := name.find("_")) != -1:
            return cls(name[0:i], name[i + 1 :], None)
        return cls(name, None, None)


@dataclass(frozen=True)
class CompartmentDef:
    """
    Defines an IPM compartment.

    Parameters
    ----------
    name :
        The name of the compartment.
    tags :
        Tags associated with the compartment.
    description :
        An optional description of the compartment.
    """

    name: CompartmentName
    """The name of the compartment."""
    tags: list[str]
    """Tags associated with the compartment."""
    description: str | None = field(default=None)
    """An optional description of the compartment."""

    def with_strata(self, strata: str) -> Self:
        """
        Return a copy of this compartment with the strata changed.

        Parameters
        ----------
        strata :
            The new strata.

        Returns
        -------
        :
            The new compartment definition.
        """
        return dataclasses.replace(self, name=self.name.with_strata(strata))


def compartment(
    name: str,
    tags: list[str] | None = None,
    description: str | None = None,
) -> CompartmentDef:
    """
    Define an IPM compartment. Convenience constructor for `CompartmentDef`.

    Parameters
    ----------
    name :
        The name of the compartment. This will be converted to a
        `CompartmentName` using the `parse` method.
    tags :
        An optional list of tags to associate with this compartment.
    description :
        An optional description of the compartment.

    Returns
    -------
    :
        The compartment definition.
    """
    return CompartmentDef(CompartmentName.parse(name), tags or [], description)


def quick_compartments(symbol_names: str) -> list[CompartmentDef]:
    """
    Define a number of IPM compartments from a space-delimited string.
    This is just short-hand syntax for the `compartment()` function.
    Note: this does not allow you to set tags or descriptions for the
    compartments.

    Parameters
    ----------
    symbol_names :
        Compartment names in a single string separated by spaces.
        For example: "S I R".

    Returns
    -------
    :
        The corresponding list of compartment definitions.
    """
    return [compartment(name) for name in symbol_names.split()]


#####################
# Model Transitions #
#####################


@dataclass(frozen=True)
class EdgeName:
    """
    The name of a transition edge, from one compartment to another.

    Parameters
    ----------
    compartment_from :
        The source compartment for this edge.
    compartment_to :
        The destination compartment for this edge.
    """

    compartment_from: CompartmentName
    """The source compartment for this edge."""
    compartment_to: CompartmentName
    """The destination compartment for this edge."""
    full: str = field(init=False, hash=False, compare=False)
    """The name of the edge as a string."""

    def __post_init__(self):
        full = f"{self.compartment_from} → {self.compartment_to}"
        object.__setattr__(self, "full", full)
        if (
            self.compartment_from.subscript != "exogenous"
            and self.compartment_to.subscript != "exogenous"
            and self.compartment_from.strata != self.compartment_to.strata
        ):
            raise ValueError(f"Edges must be within a single strata ({full})")

    def with_subscript(self, subscript: str | None) -> Self:
        """
        Return a copy of this edge with the subscript changed.

        Parameters
        ----------
        subscript :
            The new subscript.

        Returns
        -------
        :
            The new edge name.
        """
        return dataclasses.replace(
            self,
            compartment_from=self.compartment_from.with_subscript(subscript),
            compartment_to=self.compartment_to.with_subscript(subscript),
        )

    def with_strata(self, strata: str | None) -> Self:
        """
        Return a copy of this edge with the strata changed.

        Parameters
        ----------
        strata :
            The new strata.

        Returns
        -------
        :
            The new edge name.
        """
        return dataclasses.replace(
            self,
            compartment_from=self.compartment_from.with_strata(strata),
            compartment_to=self.compartment_to.with_strata(strata),
        )

    def __str__(self) -> str:
        return self.full


@dataclass(frozen=True)
class EdgeDef:
    """
    Defines a single edge transition in a compartment model.

    Parameters
    ----------
    name :
        The name of the edge.
    rate :
        The rate of flow along this edge.
    compartment_from :
        The symbol describing the source compartment.
    compartment_to :
        The symbol describing the destination compartment.
    """

    name: EdgeName
    """The name of the edge."""
    rate: Expr
    """The rate of flow along this edge."""
    compartment_from: Symbol
    """The symbol describing the source compartment."""
    compartment_to: Symbol
    """The symbol describing the destination compartment."""

    @property
    def tuple(self) -> tuple[str, str]:
        """The edge in tuple form: `(from_name, to_name)`."""
        return str(self.compartment_from), str(self.compartment_to)


def edge(
    compartment_from: Symbol,
    compartment_to: Symbol,
    rate: Expr | int | float,
) -> EdgeDef:
    """
    Define a transition edge from one compartment to another at the given rate.

    Parameters
    ----------
    compartment_from :
        The symbol describing the source compartment.
    compartment_to :
        The symbol describing the destination compartment.
    rate :
        The rate of flow along this edge.

    Returns
    -------
    :
        The edge definition.
    """
    if isinstance(rate, int):
        _rate = Integer(rate)
    elif isinstance(rate, float):
        _rate = Float(rate)
    else:
        _rate = rate
    name = EdgeName(
        CompartmentName.parse(str(compartment_from)),
        CompartmentName.parse(str(compartment_to)),
    )
    return EdgeDef(name, _rate, compartment_from, compartment_to)


@dataclass(frozen=True)
class ForkDef:
    """
    Defines a fork-style transition in a compartment model.

    Parameters
    ----------
    rate :
        The shared base-rate of the fork.
    edges :
        The edges that participate in the fork.
    probs :
        The probability of each edge in the fork.
    """

    rate: Expr
    """The shared base-rate of the fork."""
    edges: list[EdgeDef]
    """The edges that participate in the fork."""
    probs: list[Expr]
    """The probability of each edge in the fork."""

    def __str__(self) -> str:
        lhs = str(self.edges[0].compartment_from)
        rhs = ",".join([str(edge.compartment_to) for edge in self.edges])
        return f"{lhs} → ({rhs})"


def fork(*edges: EdgeDef) -> ForkDef:
    """
    Define a forked transition: a set of edges that come from the same compartment
    but go to different compartments. It is assumed the edges will share a
    "base rate"-- a common sub-expression among all edge rates --
    and that each edge in the fork is given a proportion on that base rate.

    Parameters
    ----------
    *edges :
        All edges that participate in the fork, as a var-arg.

    Returns
    -------
    :
        The fork definition.

    Examples
    --------
    Consider two edges with rates:

    1. `delta * EXPOSED * rho`
    2. `delta * EXPOSED * (1 - rho)`

    `delta * EXPOSED` is the base rate and `rho` describes the proportional split for
    each edge.
    """

    # First verify that the edges all come from the same state.
    if len(set(e.compartment_from for e in edges)) > 1:
        err = (
            "In a Fork, all edges must share the same `state_from`.\n"
            f"  Problem in: {str(edges)}"
        )
        raise IPMValidationError(err)
    # it is assumed the fork's edges are defined with complementary rate expressions
    edge_rates = [e.rate for e in edges]
    # the "base rate" -- how many individuals transition on any of these edges --
    # is the sum of all the edge rates (this defines the lambda for the poisson draw)
    rate = simplify_sum(edge_rates)
    # the probability of following a particular edge is then the edge's rate divided by
    # the base rate
    # (this defines the probability split in the eventual multinomial draw)
    probs = [simplify(r / rate) for r in edge_rates]  # type: ignore
    return ForkDef(rate, list(edges), probs)


TransitionDef = EdgeDef | ForkDef
"""The ways to define a compartment model transition: either a single edge or a fork."""


def _as_events(trxs: Iterable[TransitionDef]) -> Iterator[EdgeDef]:
    """
    Return an iterator for all unique events defined in the transition model.
    Each edge corresponds to a single event, even the edges that are part of a fork.
    The events are returned in a stable order (definition order) so that they can be
    indexed that way.
    """
    for t in trxs:
        match t:
            case EdgeDef() as e:
                yield e
            case ForkDef(_, edges):
                for e in edges:
                    yield e


def _remap_edge(
    e: EdgeDef,
    strata: str,
    symbol_mapping: dict[Symbol, Symbol],
) -> EdgeDef:
    return EdgeDef(
        name=e.name.with_strata(strata),
        rate=substitute(e.rate, symbol_mapping),
        compartment_from=symbol_mapping[e.compartment_from],
        compartment_to=symbol_mapping[e.compartment_to],
    )


def _remap_fork(
    f: ForkDef,
    strata: str,
    symbol_mapping: dict[Symbol, Symbol],
) -> ForkDef:
    return ForkDef(
        rate=substitute(f.rate, symbol_mapping),
        edges=[_remap_edge(e, strata, symbol_mapping) for e in f.edges],
        probs=[substitute(p, symbol_mapping) for p in f.probs],
    )


def _remap_transition(
    t: TransitionDef,
    strata: str,
    symbol_mapping: dict[Symbol, Symbol],
) -> TransitionDef:
    """Replace symbols in the transition using substitution from `symbol_mapping`."""
    match t:
        case EdgeDef():
            return _remap_edge(t, strata, symbol_mapping)
        case ForkDef():
            return _remap_fork(t, strata, symbol_mapping)


######################
# Compartment Models #
######################


class ModelSymbols:
    """
    IPM symbols needed in defining the model's transition rate expressions.

    Parameters
    ----------
    compartments :
        The compartments of the IPM, as name/symbolic name pairs.
    requirements :
        The requirements (data attributes) of the IPM, as name/symbolic name pairs.
    """

    all_compartments: Sequence[Symbol]
    """Compartment symbols in definition order."""
    all_requirements: Sequence[Symbol]
    """Requirements symbols in definition order."""

    _csymbols: dict[str, Symbol]
    """Mapping of compartment name to symbol."""
    _rsymbols: dict[str, Symbol]
    """Mapping of requirement name to symbol."""

    def __init__(
        self,
        compartments: Sequence[tuple[str, str]],
        requirements: Sequence[tuple[str, str]],
    ):
        # NOTE: the arguments here are tuples of name and symbolic name;
        # this is redundant for single-strata models, but allows multistrata models
        # to keep fine-grained control over symbol substitution while allowing
        # the user to refer to the names they already know.
        cs = [(n, to_symbol(s)) for n, s in compartments]
        rs = [(n, to_symbol(s)) for n, s in requirements]
        self.all_compartments = [s for _, s in cs]
        self.all_requirements = [s for _, s in rs]
        self._csymbols = dict(cs)
        self._rsymbols = dict(rs)

    def compartments(self, *names: str) -> Sequence[Symbol]:
        """
        Select compartment symbols by name.

        Parameters
        ----------
        *names :
            The names of the model's compartments to select.

        Returns
        -------
        :
            The symbols representing the compartments in the order in which they're
            named. Ideal for unpacking into variables.

        Examples
        --------
        >>> [S, I, R] = symbols.compartments("S", "I", "R")
        >>> print(f"{type(S)}: {S}")
        <class 'sympy.core.symbol.Symbol'>: S
        """
        return [self._csymbols[n] for n in names]

    def requirements(self, *names: str) -> Sequence[Symbol]:
        """
        Select requirement symbols by name.

        Parameters
        ----------
        *names :
            The names of the model's attributes to select.

        Returns
        -------
        :
            The symbols representing the attributes in the order in which they're named.
            Ideal for unpacking into variables.

        Examples
        --------
        >>> [alpha, beta, gamma] = symbols.requirements("alpha", "beta", "gamma")
        >>> print(f"{type(alpha)}: {alpha}")
        <class 'sympy.core.symbol.Symbol'>: alpha
        """
        return [self._rsymbols[n] for n in names]


class BaseCompartmentModel(ABC):
    """
    Shared base-class for compartment models.

    See Also
    --------
    In practice users will mostly use [epymorph.compartment_model.CompartmentModel][]
    for single-strata IPMs, and construct multi-strata IPMs as a byproduct of
    constructing a multi-strata RUME.
    """

    compartments: Sequence[CompartmentDef] = ()
    """The compartments of the model."""

    requirements: Sequence[AttributeDef] = ()
    """The attributes required by the model."""

    # NOTE: these two (above) attributes are coded as such so that overriding
    # this class is simpler for users. Normally I'd make them properties,
    # -- since they really should not be modified after creation --
    # but this would increase the implementation complexity.
    # And to avoid requiring users to call the initializer, the rest
    # of the attributes are cached_properties which initialize lazily.

    # These private attributes will be computed during instance initialization
    # by the metaclass. It's better to do eager evaluation of these things
    # so that issues in model construction pop up sooner than later.

    @property
    def quantities(self) -> Iterator[CompartmentDef | EdgeDef]:
        """All quantities in the model, compartments then edges, in definition order."""
        yield from self.compartments
        yield from self.events

    @property
    @abstractmethod
    def symbols(self) -> ModelSymbols:
        """The symbols which represent parts of this model."""

    @property
    @abstractmethod
    def transitions(self) -> Sequence[TransitionDef]:
        """The transitions in the model."""

    @property
    @abstractmethod
    def events(self) -> Sequence[EdgeDef]:
        """The unique transition events in the model."""

    @property
    @abstractmethod
    def requirements_dict(self) -> OrderedDict[AbsoluteName, AttributeDef]:
        """The attributes required by this model."""

    @property
    def num_compartments(self) -> int:
        """The number of compartments in this model."""
        return len(self.compartments)

    @property
    def num_events(self) -> int:
        """The number of distinct events (transitions) in this model."""
        return len(self.events)

    @property
    @abstractmethod
    def strata(self) -> Sequence[str]:
        """The names of the strata involved in this compartment model."""

    @property
    @abstractmethod
    def is_multistrata(self) -> bool:
        """True if this compartment model is multistrata, False for single-strata."""

    @property
    def select(self) -> "QuantitySelector":
        """Make a quantity selection from this IPM."""
        return QuantitySelector(self)

    def diagram(
        self,
        *,
        file: str | Path | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> None:
        """
        Render a diagram of this IPM, either by showing it with matplotlib (default)
        or by saving it to `file` as a png image.

        Parameters
        ----------
        file :
            Provide a file path to save a png image of the diagram to this path.
            If `file` is None, we will instead use matplotlib to show the diagram.
        figsize :
            The matplotlib figure size to use when displaying the diagram.
            Only used if `file` is not provided.
        """
        render_diagram(ipm=self, file=file, figsize=figsize)


def validate_compartment_model(model: BaseCompartmentModel) -> None:
    """
    Validate an IPM definition.

    Parameters
    ----------
    model :
        The IPM to validate.

    Raises
    ------
    IPMValidationError
        If there are structural issues in the IPM.
    """
    name = model.__class__.__name__

    # we need a sneaky way to suppress IPM validation warnings sometimes
    suppress_warnings = getattr(model, "_suppress_ipm_validation_warnings", False)

    # transitions cannot have the source and destination both be exogenous;
    # this would be madness.
    if any(
        edge.compartment_from in exogenous_states
        and edge.compartment_to in exogenous_states
        for edge in model.events
    ):
        err = (
            f"Invalid transitions in {name}: "
            "transitions cannot use exogenous states (BIRTH/DEATH) "
            "as both source and destination."
        )
        raise IPMValidationError(err)

    # All symbols used in rate expressions, including compartments.
    rate_symbols = set(
        symbol
        for e in model.events
        for symbol in e.rate.free_symbols
        if isinstance(symbol, Symbol)
    )

    # Compartments used as either a 'from' or a 'to' in a transition.
    trx_comps = set(
        compartment
        for e in model.events
        for compartment in [e.compartment_from, e.compartment_to]
        # don't include exogenous states in the compartment set
        if compartment not in exogenous_states
    )

    # All symbols used anywhere in transitions (from + to + rate expr)
    trx_symbols = rate_symbols.union(trx_comps)

    # Extract the set of requirements used by transition rate expressions
    # by taking all used symbols and subtracting compartment symbols.
    trx_reqs = rate_symbols.difference(model.symbols.all_compartments)

    # transition compartments minus declared compartments should be empty
    missing_comps = trx_comps.difference(model.symbols.all_compartments)
    if len(missing_comps) > 0:
        err = (
            f"Invalid transitions in {name}: "
            "transitions reference compartments which were not declared.\n"
            f"Missing compartments: {', '.join(map(str, missing_comps))}"
        )
        raise IPMValidationError(err)

    # declared compartments minus used compartments is ideally empty,
    # otherwise raise a warning
    if not suppress_warnings:
        extra_comps = set(model.symbols.all_compartments).difference(trx_symbols)
        if len(extra_comps) > 0:
            msg = (
                f"Possible issue in {name}: "
                "not all declared compartments are being used in transitions.\n"
                f"Extra compartments: {', '.join(map(str, extra_comps))}"
            )
            warn(msg)

    # transition requirements minus declared requirements should be empty
    missing_reqs = trx_reqs.difference(model.symbols.all_requirements)
    if len(missing_reqs) > 0:
        err = (
            f"Invalid transitions in {name}: "
            "transitions reference requirements which were not declared.\n"
            f"Missing requirements: {', '.join(map(str, missing_reqs))}"
        )
        raise IPMValidationError(err)

    # declared requirements minus used requirements is ideally empty,
    # otherwise raise a warning
    if not suppress_warnings:
        extra_reqs = set(model.symbols.all_requirements).difference(trx_reqs)
        if len(extra_reqs) > 0:
            msg = (
                f"Possible issue in {name}: "
                "not all declared requirements are being used in transitions.\n"
                f"Extra requirements: {', '.join(map(str, extra_reqs))}"
            )
            warn(msg)


####################################
# Single-strata Compartment Models #
####################################


class CompartmentModelClass(ABCMeta):
    """`CompartmentModel` metaclass; enforces proper implementation."""

    def __new__(
        cls: Type["CompartmentModelClass"],
        name: str,
        bases: tuple[type, ...],
        dct: dict[str, Any],
    ) -> "CompartmentModelClass":
        # Skip these checks for abstract classes:
        cls0 = super().__new__(cls, name, bases, dct)
        if getattr(cls0, "__abstractmethods__", False):
            return cls0

        # Check model compartments.
        cmps = dct.get("compartments")
        if cmps is None or not isinstance(cmps, (list, tuple)):
            err = f"Invalid compartments in {name}: please specify as a list or tuple."
            raise IPMValidationError(err)
        if len(cmps) == 0:
            err = (
                f"Invalid compartments in {name}: "
                "please specify at least one compartment."
            )
            raise IPMValidationError(err)
        if not are_instances(cmps, CompartmentDef):
            err = (
                f"Invalid compartments in {name}: must be instances of CompartmentDef."
            )
            raise IPMValidationError(err)
        if not are_unique(c.name for c in cmps):
            err = f"Invalid compartments in {name}: compartment names must be unique."
            raise IPMValidationError(err)
        # Make compartments immutable.
        dct["compartments"] = tuple(cmps)

        # Check transitions... we have to instantiate the class.
        new_cls = super().__new__(cls, name, bases, dct)
        instance = new_cls()
        validate_compartment_model(instance)
        return new_cls

    def __call__(cls, *args, **kwargs):
        # Perform our initialization on all newly created instances.
        # This allows us to bypass writing an __init__ function, which
        # end users would then have to remember to call when subclassing.
        # This should make implementations easier to write.
        instance = super().__call__(*args, **kwargs)
        instance._construct_model()
        return instance


class CompartmentModel(BaseCompartmentModel, ABC, metaclass=CompartmentModelClass):
    """
    A compartment model definition and its corresponding metadata.
    Effectively, a collection of compartments, transitions between compartments,
    and the data parameters which are required to compute the transitions.

    Examples
    --------
    --8<-- "docs/_examples/compartment_model_CompartmentModel.md"
    """

    @abstractmethod
    def edges(self, symbols: ModelSymbols) -> Sequence[TransitionDef]:
        """
        When implementing a `CompartmentModel`, override this method
        to define the transition edges between compartments.

        Parameters
        ----------
        symbols :
            An object containing the symbols in the model for use in
            declaring edges. These include compartments and data requirements.

        Returns
        -------
        :
            The transitions for the model.
        """

    _symbols: ModelSymbols
    _transitions: Sequence[TransitionDef]
    _events: Sequence[EdgeDef]
    _requirements_dict: OrderedDict[AbsoluteName, AttributeDef]

    @final
    def _construct_model(self):
        # epymorph's initialization logic, invoked by the metaclass
        # (see metaclass __call__ for more info)
        self._symbols = ModelSymbols(
            [(c.name.full, c.name.full) for c in self.compartments],
            [(r.name, r.name) for r in self.requirements],
        )
        self._transitions = self.edges(self.symbols)
        self._events = list(_as_events(self._transitions))
        self._requirements_dict = OrderedDict(
            [
                (AbsoluteName(gpm_strata(DEFAULT_STRATA), "ipm", r.name), r)
                for r in self.requirements
            ]
        )

    @property
    @override
    def symbols(self) -> ModelSymbols:
        return self._symbols

    @property
    @override
    def transitions(self) -> Sequence[TransitionDef]:
        return self._transitions

    @property
    @override
    def events(self) -> Sequence[EdgeDef]:
        return self._events

    @property
    @override
    def requirements_dict(self) -> OrderedDict[AbsoluteName, AttributeDef]:
        return self._requirements_dict

    @property
    @override
    def strata(self) -> Sequence[str]:
        return ["all"]

    @property
    @override
    def is_multistrata(self) -> bool:
        return False


###################################
# Multi-strata Compartment Models #
###################################


class MultiStrataModelSymbols(ModelSymbols):
    """
    IPM symbols needed in defining the model's transition rate expressions.

    Parameters
    ----------
    strata :
        The strata and the compartment model for each.
    meta_requirements :
        Additional requirements needed for the combined model's meta-transitions.
    """

    all_meta_requirements: Sequence[Symbol]
    """Meta-requirement symbols in definition order."""

    _msymbols: dict[str, Symbol]
    """Mapping of meta requirements name to symbol."""

    strata: Sequence[str]
    """The strata names used in this model."""

    _strata_symbols: dict[str, ModelSymbols]
    """
    Mapping of strata name to the symbols of that strata.
    The symbols within use their original names.
    """

    def __init__(
        self,
        strata: Sequence[tuple[str, CompartmentModel]],
        meta_requirements: Sequence[AttributeDef],
    ):
        # These are all tuples of:
        # (original name, strata name, symbolic name)
        # where the symbolic name is disambiguated by appending
        # the strata it belongs to.
        cs = [
            (c.name.full, strata_name, f"{c.name}_{strata_name}")
            for strata_name, ipm in strata
            for c in ipm.compartments
        ]
        rs = [
            (r.name, strata_name, f"{r.name}_{strata_name}")
            for strata_name, ipm in strata
            for r in ipm.requirements
        ]
        ms = [(r.name, "meta", f"{r.name}_meta") for r in meta_requirements]

        super().__init__(
            compartments=[(sym, sym) for _, _, sym in cs],
            requirements=[
                *((sym, sym) for _, _, sym in rs),
                *((orig, sym) for orig, _, sym in ms),
            ],
        )

        self.strata = [strata_name for strata_name, _ in strata]
        self._strata_symbols = {
            strata_name: ModelSymbols(
                compartments=[
                    (orig, sym) for orig, strt, sym in cs if strt == strata_name
                ],
                requirements=[
                    (orig, sym) for orig, strt, sym in rs if strt == strata_name
                ],
            )
            for strata_name, _ in strata
        }

        self.all_meta_requirements = [to_symbol(sym) for _, _, sym in ms]
        self._msymbols = {orig: to_symbol(sym) for orig, _, sym in ms}

    def strata_compartments(self, strata: str, *names: str) -> Sequence[Symbol]:
        """
        Select compartment symbols by name in a particular strata.
        If `names` is non-empty, select those symbols by their original name.
        If `names` is empty, return all symbols.

        Parameters
        ----------
        strata :
            The strata to select.
        *names :
            The names of the model's compartments to select, or left empty to select all
            compartments in the strata.

        Returns
        -------
        :
            The symbols representing the compartments in the order in which they're
            named, or their definition order if selecting all. Ideal for unpacking into
            variables.
        """
        sym = self._strata_symbols[strata]
        return sym.all_compartments if len(names) == 0 else sym.compartments(*names)

    def strata_requirements(self, strata: str, *names: str) -> Sequence[Symbol]:
        """
        Select requirement symbols by name in a particular strata.
        If `names` is non-empty, select those symbols by their original name.
        If `names` is empty, return all symbols.

        Parameters
        ----------
        strata :
            The strata to select.
        *names :
            The names of the model's requirements to select, or left empty to select all
            requirements for the strata.

        Returns
        -------
        :
            The symbols representing the attributes in the order in which they're
            named, or their definition order if selecting all. Ideal for unpacking into
            variables.
        """
        sym = self._strata_symbols[strata]
        return sym.all_requirements if len(names) == 0 else sym.requirements(*names)


MetaEdgeBuilder = Callable[[MultiStrataModelSymbols], Sequence[TransitionDef]]
"""The type of a function for creating meta edges in a multistrata RUME."""


class CombinedCompartmentModel(BaseCompartmentModel):
    """
    A `CompartmentModel` constructed by combining others for use in multi-strata models.
    Typically you will not have to create a `CombinedCompartmentModel` directly.
    Building a `MultiStrataRUME` will combine the models for you.

    Parameters
    ----------
    strata :
        A list of pairs mapping strata names to the IPM for the strata.
    meta_requirements :
        Additional requirement definitions for use in the combined model's meta edges.
    meta_edges :
        A function which defines the combined model's meta edges.
    """

    compartments: Sequence[CompartmentDef]
    """All compartments; renamed with strata."""
    requirements: Sequence[AttributeDef]
    """All requirements, including meta-requirements."""

    _strata: Sequence[tuple[str, CompartmentModel]]
    _meta_requirements: Sequence[AttributeDef]
    _meta_edges: MetaEdgeBuilder
    _symbols: MultiStrataModelSymbols
    _transitions: Sequence[TransitionDef]
    _events: Sequence[EdgeDef]
    _requirements_dict: OrderedDict[AbsoluteName, AttributeDef]

    def __init__(
        self,
        strata: Sequence[tuple[str, CompartmentModel]],
        meta_requirements: Sequence[AttributeDef],
        meta_edges: MetaEdgeBuilder,
    ):
        self._strata = strata
        self._meta_requirements = meta_requirements
        self._meta_edges = meta_edges

        self.compartments = [
            comp.with_strata(strata_name)
            for strata_name, ipm in strata
            for comp in ipm.compartments
        ]

        self.requirements = [
            *(r for _, ipm in strata for r in ipm.requirements),
            *self._meta_requirements,
        ]

        symbols = MultiStrataModelSymbols(
            strata=self._strata, meta_requirements=self._meta_requirements
        )
        self._symbols = symbols

        # Figure out the per-strata mapping from old symbol to new symbol
        # by matching everything up in-order.
        strata_mapping = list[dict[Symbol, Symbol]]()
        # And a mapping from new (stratified) symbols back to their original form
        # and which strata they belong to.
        reverse_mapping = dict[Symbol, tuple[str | None, Symbol]]()
        all_cs = iter(symbols.all_compartments)
        all_rs = iter(symbols.all_requirements)
        for strata_name, ipm in self._strata:
            mapping = {x: x for x in exogenous_states}
            old = ipm.symbols
            for old_symbol in old.all_compartments:
                new_symbol = next(all_cs)
                mapping[old_symbol] = new_symbol
                reverse_mapping[new_symbol] = (strata_name, old_symbol)
            for old_symbol in old.all_requirements:
                new_symbol = next(all_rs)
                mapping[old_symbol] = new_symbol
                reverse_mapping[new_symbol] = (strata_name, old_symbol)
            strata_mapping.append(mapping)
        # (exogenous states just map to themselves, no strata)
        reverse_mapping |= {x: (None, x) for x in exogenous_states}

        # The meta_edges function produces edges with invalid names:
        # users `edge()` which just parses the symbol string, but this causes
        # the strata to be mistaken as a subscript. This function fixes things.
        def fix_edge_names(x: TransitionDef) -> TransitionDef:
            match x:
                case ForkDef():
                    edges = [fix_edge_names(e) for e in x.edges]
                    return dataclasses.replace(x, edges=edges)
                case EdgeDef():
                    s_from, c_from = reverse_mapping[x.compartment_from]
                    s_to, c_to = reverse_mapping[x.compartment_to]
                    strata = next(s for s in (s_from, s_to) if s is not None)
                    name = EdgeName(
                        CompartmentName.parse(str(c_from)),
                        CompartmentName.parse(str(c_to)),
                    ).with_strata(strata)
                    return dataclasses.replace(x, name=name)

        self._transitions = [
            *(
                _remap_transition(trx, strata, mapping)
                for (strata, ipm), mapping in zip(self._strata, strata_mapping)
                for trx in ipm.transitions
            ),
            *(fix_edge_names(x) for x in self._meta_edges(symbols)),
        ]
        self._events = list(_as_events(self._transitions))
        self._requirements_dict = OrderedDict(
            [
                *(
                    (AbsoluteName(gpm_strata(strata_name), "ipm", r.name), r)
                    for strata_name, ipm in self._strata
                    for r in ipm.requirements
                ),
                *(
                    (AbsoluteName(META_STRATA, "ipm", r.name), r)
                    for r in self._meta_requirements
                ),
            ]
        )

    @property
    @override
    def symbols(self) -> MultiStrataModelSymbols:
        return self._symbols

    @property
    @override
    def transitions(self) -> Sequence[TransitionDef]:
        return self._transitions

    @property
    @override
    def events(self) -> Sequence[EdgeDef]:
        return self._events

    @property
    @override
    def requirements_dict(self) -> OrderedDict[AbsoluteName, AttributeDef]:
        return self._requirements_dict

    @property
    @override
    def strata(self) -> Sequence[str]:
        return [name for name, _ in self._strata]

    @property
    @override
    def is_multistrata(self) -> bool:
        return True


#####################################################
# Compartment Model quantity select/group/aggregate #
#####################################################

Quantity = CompartmentDef | EdgeDef
"""
The type of quantity referenced by a `QuantityStrategy`, either compartments or events.
"""


class QuantityGroupResult(NamedTuple):
    """The result of a quantity grouping operation."""

    groups: tuple[Quantity, ...]
    """The quantities (or psuedo-quantities) representing each group."""
    indices: tuple[tuple[int, ...], ...]
    """The IPM quantity indices included in each group."""


_N = TypeVar("_N", bound=CompartmentName | EdgeName)


class QuantityGrouping(NamedTuple):
    """
    Describes how to group simulation output quantities (events and compartments).
    The default combines any quantity whose names match exactly. This is common in
    multistrata models where events from several strata impact one transition.
    You can also choose to group across strata and subscript differences.
    Setting `strata` or `subscript` to True means those elements of quantity names
    (if they exist) are ignored for the purposes of matching.
    """

    strata: bool
    """True to combine quantities across strata."""
    subscript: bool
    """True to combine quantities across subscript."""

    def _strip(self, name: _N) -> _N:
        if self.strata:
            name = name.with_strata(None)
        if self.subscript:
            name = name.with_subscript(None)
        return name

    def map(self, quantities: Sequence[Quantity]) -> QuantityGroupResult:
        """
        Perform the quantity grouping.

        Parameters
        ----------
        quantities :
            The quantities to subject to grouping.

        Returns
        -------
        :
            The grouping result.
        """
        # first simplify the names to account for `strata` and `subscript`
        names = [self._strip(q.name) for q in quantities]
        # the groups are now the unique names in the list (maintain ordering)
        group_names = filter_unique(names)
        # figure out which original quantities belong in each group (by index)
        group_indices = tuple(
            tuple(j for j, qty in enumerate(names) if group == qty)
            for group in group_names
        )

        # we can create an artificial CompartmentDef or EdgeDef for each group
        # if we assume compartments and events will never mix (which they shouldn't)
        def _combine(
            group_name: CompartmentName | EdgeName,
            indices: tuple[int, ...],
        ) -> Quantity:
            qs = [q for i, q in enumerate(quantities) if i in indices]
            if isinstance(group_name, CompartmentName) and are_instances(
                qs, CompartmentDef
            ):
                return CompartmentDef(group_name, [], None)
            elif isinstance(group_name, EdgeName) and are_instances(qs, EdgeDef):
                return EdgeDef(
                    name=group_name,
                    rate=Add(*[q.rate for q in qs]),
                    compartment_from=to_symbol(group_name.compartment_from.full),
                    compartment_to=to_symbol(group_name.compartment_to.full),
                )
            # If we got here, it probably means compartments and groups wound
            # up in the same group somehow. This should not be possible,
            # so something went terribly wrong.
            raise ValueError("Unable to compute quantity groups.")

        groups = tuple(_combine(n, i) for n, i in zip(group_names, group_indices))
        return QuantityGroupResult(groups, group_indices)


QuantityAggMethod = Literal["sum"]
"""The supported methods for aggregating IPM quantities."""


@dataclass(frozen=True)
class QuantityStrategy:
    """
    A strategy for dealing with the quantity axis, e.g., in processing results.
    Quantities here are an IPM's compartments and events. Strategies can include
    selection of a subset, grouping, and aggregation.

    Typically you will create one of these by calling methods on a `QuantitySelector`
    instance.

    Parameters
    ----------
    ipm :
        The original IPM quantity information.
    selection :
        A boolean mask for selection of a subset of quantities.
    grouping :
        A method for grouping IPM quantities.
    aggregation :
        A method for aggregating the quantity groups.
    """

    ipm: BaseCompartmentModel
    """The original IPM quantity information."""
    selection: NDArray[np.bool_]
    """A boolean mask for selection of a subset of quantities."""
    grouping: QuantityGrouping | None
    """A method for grouping IPM quantities."""
    aggregation: QuantityAggMethod | None
    """A method for aggregating the quantity groups."""

    @property
    def selected(self) -> Sequence[Quantity]:
        """The quantities from the IPM which are selected, prior to any grouping."""
        return [q for sel, q in zip(self.selection, self.ipm.quantities) if sel]

    @property
    @abstractmethod
    def quantities(self) -> Sequence[Quantity]:
        """
        The quantities in the result. If the strategy performs grouping these
        may be pseudo-quantities made by combining the quantities in the group.
        """

    @property
    @abstractmethod
    def labels(self) -> Sequence[str]:
        """Labels for the quantities in the result, after any grouping."""

    def disambiguate(self) -> OrderedDict[str, str]:
        """
        Create a name mapping to disambiguate IPM quantities that have
        the same name. This happens commonly in multistrata IPMs with
        meta edges where multiple other strata influence a transmission rate
        in a single strata. The returned mapping includes only the selected IPM
        compartments and events, but definition order is maintained.
        Keys are the unique name and values are the original names
        (because the original names might contain duplicates);
        so you will have to map into unique names by position, but can map
        back using this mapping directly.

        This function is intended for epymorph's internal use.
        """
        selected = [
            (i, q) for i, q in enumerate(self.ipm.quantities) if self.selection[i]
        ]
        qs_original = [q.name.full for i, q in selected]
        qs_renamed = [f"{q.name}_{i}" for i, q in selected]
        return OrderedDict(zip(qs_renamed, qs_original))

    def disambiguate_groups(self) -> OrderedDict[str, str]:
        """
        Like method `disambiguate` but for working with quantities
        after any grouping has been performed. If grouping is None,
        this is equivalent to `disambiguate`.

        This function is intended for epymorph's internal use.
        """
        if self.grouping is None:
            return self.disambiguate()
        groups, _ = self.grouping.map(self.selected)
        selected = [(i, q) for i, q in enumerate(groups)]
        qs_original = [q.name.full for i, q in selected]
        qs_renamed = [f"{q.name}_{i}" for i, q in selected]
        return OrderedDict(zip(qs_renamed, qs_original))


@dataclass(frozen=True)
class QuantitySelection(QuantityStrategy):
    """
    A kind of `QuantityStrategy` describing a sub-selection of IPM quantities
    (its events and compartments). A selection performs no grouping or aggregation.

    Typically you will create one of these by calling methods on `QuantitySelector`
    instance.

    Parameters
    ----------
    ipm :
        The original IPM quantities information.
    selection :
        A boolean mask for selection of a subset of IPM quantities.
    """

    ipm: BaseCompartmentModel
    """The original IPM quantities information."""
    selection: NDArray[np.bool_]
    """A boolean mask for selection of a subset of IPM quantities."""
    grouping: None = field(init=False, default=None)
    """A method for grouping IPM quantities."""
    aggregation: None = field(init=False, default=None)
    """A method for aggregating the quantity groups."""

    @property
    @override
    def quantities(self) -> Sequence[CompartmentDef | EdgeDef]:
        return self.selected

    @property
    @override
    def labels(self) -> Sequence[str]:
        return [q.name.full for q in self.selected]

    @property
    def compartment_index(self) -> int:
        """
        The selected compartment index, if and only if there is exactly
        one compartment in the selection. Otherwise, raises ValueError.

        See `compartment_indices` if you want to select possibly multiple indices.
        """
        indices = self.compartment_indices
        if len(indices) != 1:
            err = (
                "Your selection must contain exactly one compartment to use this "
                "method. Use `compartment_indices` if you want to select more."
            )
            raise ValueError(err)
        return indices[0]

    @property
    def compartment_indices(self) -> tuple[int, ...]:
        """
        The selected compartment indices. These indices may be useful
        for instance to access a simulation output's `compartments` result array.
        May be an empty tuple.
        """
        C = self.ipm.num_compartments
        return tuple(i for i in np.flatnonzero(self.selection) if i < C)

    @property
    def event_index(self) -> int:
        """
        The selected event index, if and only if there is exactly
        one event in the selection. Otherwise, raises ValueError.

        See `event_indices` if you want to select possibly multiple indices.
        """
        indices = self.event_indices
        if len(indices) != 1:
            err = (
                "Your selection must contain exactly one event to use this "
                "method. Use `event_indices` if you want to select more."
            )
            raise ValueError(err)
        return indices[0]

    @property
    def event_indices(self) -> tuple[int, ...]:
        """
        The selected event indices. These indices may be useful
        for instance to access a simulation output's `events` result array.
        May be an empty tuple.
        """
        C = self.ipm.num_compartments
        return tuple(i - C for i in np.flatnonzero(self.selection) if i >= C)

    def group(
        self,
        *,
        strata: bool = False,
        subscript: bool = False,
    ) -> "QuantityGroup":
        """
        Group quantities according to the given options.

        By default, any quantities that directly match each other will be combined.
        This generally only happens with events, where there may be multiple edges
        between the same compartments like `S->I`, perhaps due to meta edges in a
        multistrata model.

        With `strata=True`, quantities that would match if you removed the strata name
        will be combined. e.g., `S_young` and `S_old`;
        or `S_young->I_young` and `S_old->I_old`.

        With `subscript=True`, quantities that would match if you removed subscript
        names will be combined. e.g., `I_asymptomatic_young` and `I_symptomatic_young`
        belong to the same strata (young) but have different subscripts so they will
        be combined.

        And if both options are True, we consider matches after removing both strata
        and subscript names -- effectively matching on the base compartment and
        event names.

        Parameters
        ----------
        strata :
            Whether or not to combine different strata.
        subscript :
            Whether or not the combine different subscripts.

        Returns
        -------
        :
            A grouping strategy object.
        """
        return QuantityGroup(
            self.ipm,
            self.selection,
            QuantityGrouping(strata, subscript),
        )


@dataclass(frozen=True)
class QuantityGroup(QuantityStrategy):
    """
    A kind of `QuantityStrategy` describing a group operation on IPM quantities, with
    an optional sub-selection.

    Typically you will create one of these by calling methods on a `QuantitySelection`
    instance.

    Parameters
    ----------
    ipm :
        The original IPM quantity information.
    selection :
        A boolean mask for selection of a subset of quantities.
    grouping :
        A method for grouping IPM quantities.
    """

    ipm: BaseCompartmentModel
    """The original IPM quantity information."""
    selection: NDArray[np.bool_]
    """A boolean mask for selection of a subset of quantities."""
    grouping: QuantityGrouping
    """A method for grouping IPM quantities."""
    aggregation: None = field(init=False, default=None)
    """A method for aggregating the quantity groups."""

    @property
    @override
    def quantities(self) -> Sequence[CompartmentDef | EdgeDef]:
        groups, _ = self.grouping.map(self.selected)
        return groups

    @property
    @override
    def labels(self) -> Sequence[str]:
        groups, _ = self.grouping.map(self.selected)
        return [g.name.full for g in groups]

    def agg(self, agg: QuantityAggMethod) -> "QuantityAggregation":
        """
        Combine grouped quantities using the named aggregation.

        Parameters
        ----------
        agg :
            The aggregation method to use.

        Returns
        -------
        :
            The aggregation strategy object.
        """
        return QuantityAggregation(self.ipm, self.selection, self.grouping, agg)

    def sum(self) -> "QuantityAggregation":
        """
        Combine grouped quantities by adding their values. Equivalent to `agg("sum")`.

        Returns
        -------
        :
            The aggregation strategy object.
        """
        return self.agg("sum")


@dataclass(frozen=True)
class QuantityAggregation(QuantityStrategy):
    """
    A kind of `QuantityStrategy` describing a group-and-aggregate operation on IPM
    quantities, with an optional sub-selection.

    Typically you will create one of these by calling methods on a `QuantitySelector`
    instance.

    Parameters
    ----------
    ipm :
        The original IPM quantity information.
    selection :
        A boolean mask for selection of a subset of quantities.
    grouping :
        A method for grouping IPM quantities.
    aggregation :
        A method for aggregating the quantity groups.
    """

    ipm: BaseCompartmentModel
    """The original IPM quantity information."""
    selection: NDArray[np.bool_]
    """A boolean mask for selection of a subset of quantities."""
    grouping: QuantityGrouping
    """A method for grouping IPM quantities."""
    aggregation: QuantityAggMethod
    """A method for aggregating the quantity groups."""

    @property
    @override
    def quantities(self) -> Sequence[CompartmentDef | EdgeDef]:
        groups, _ = self.grouping.map(self.selected)
        return groups

    @property
    @override
    def labels(self) -> Sequence[str]:
        groups, _ = self.grouping.map(self.selected)
        return [g.name.full for g in groups]

    # NOTE: we don't support agg without a group in this axis
    # It's not really useful to squash everything together typically.


class QuantitySelector:
    """
    A utility class for selecting a subset of IPM quantities. Most of the time you
    obtain one of these using `CompartmentModel`'s `select` property.
    """

    _ipm: BaseCompartmentModel
    """The original IPM quantity information."""

    def __init__(self, ipm: BaseCompartmentModel):
        self._ipm = ipm

    def _mask(
        self,
        compartments: bool | list[bool] = False,
        events: bool | list[bool] = False,
    ) -> NDArray[np.bool_]:
        C = self._ipm.num_compartments
        E = self._ipm.num_events
        m = np.zeros(shape=C + E, dtype=np.bool_)
        if compartments is not False:
            m[:C] = compartments
        if events is not False:
            m[C:] = events
        return m

    def all(self) -> "QuantitySelection":
        """
        Select all compartments and events.

        Returns
        -------
        :
            The selection object.
        """
        m = self._mask()
        m[:] = True
        return QuantitySelection(self._ipm, m)

    def indices(self, *indices: int) -> "QuantitySelection":
        """
        Select quantities by index (determined by IPM definition order:
        all IPM compartments, all IPM events, and then meta edge events if any).

        Parameters
        ----------
        *indices :
            The indices to select, as a var-arg.

        Returns
        -------
        :
            The selection object.
        """
        m = self._mask()
        m[indices] = True
        return QuantitySelection(self._ipm, m)

    def _compile_pattern(self, pattern: str) -> re.Pattern:
        """Turn a pattern string (which is custom syntax) into a regular expression."""
        # We're not interpreting pattern as a regex directly, so escape any
        # special characters. Then replace '*' with the necessary regex.
        return re.compile(re.escape(pattern).replace(r"\*", ".*?"))

    def _compile_event_pattern(self, pattern: str) -> tuple[re.Pattern, re.Pattern]:
        """
        Interpret a pattern string as two patterns matching against the
        source and destination compartments.
        """
        try:
            # Users can use any of these options for the separator.
            if "->" in pattern:
                src, dst = pattern.split("->")
            elif "-" in pattern:
                src, dst = pattern.split("-")
            elif ">" in pattern:
                src, dst = pattern.split(">")
            else:
                err = f"Invalid event pattern syntax: {pattern}"
                raise ValueError(err)
            return (
                self._compile_pattern(src),
                self._compile_pattern(dst),
            )
        except ValueError:
            err = f"Invalid event pattern syntax: {pattern}"
            raise ValueError(err) from None

    def by(
        self,
        *,
        compartments: str | Iterable[str] = (),
        events: str | Iterable[str] = (),
    ) -> "QuantitySelection":
        """
        Select compartments and events by providing pattern strings for each.

        Providing an empty sequence implies selecting none of that type.
        Multiple patterns are combined as though by boolean-or.

        Parameters
        ----------
        compartments :
            The compartment selection patterns.
        events :
            The event selection patterns.

        Returns
        -------
        :
            The selection object.
        """
        cs = [compartments] if isinstance(compartments, str) else [*compartments]
        es = [events] if isinstance(events, str) else [*events]
        c_mask = self._mask() if len(cs) == 0 else self.compartments(*cs).selection
        e_mask = self._mask() if len(es) == 0 else self.events(*es).selection
        return QuantitySelection(self._ipm, c_mask | e_mask)

    def compartments(self, *patterns: str) -> "QuantitySelection":
        """
        Select compartments with zero or more pattern strings.

        Specify no patterns to select all compartments.
        Pattern strings match against compartment names.

        Multiple patterns are combined as though by boolean-or.
        Pattern strings can use asterisk as a wildcard character
        to match any (non-empty) part of a name besides underscores.
        For example, `"I_*"` would match events `"I_abc"` and `"I_def"`.

        Parameters
        ----------
        *patterns :
            The compartment selection patterns, as a var-arg.

        Returns
        -------
        :
            The selection object.
        """
        if len(patterns) == 0:
            # select all compartments
            mask = self._mask(compartments=True)
        else:
            mask = self._mask()
            for p in patterns:
                regex = self._compile_pattern(p)
                curr = self._mask(
                    compartments=[
                        regex.fullmatch(c.name.full) is not None
                        for c in self._ipm.compartments
                    ]
                )
                if not np.any(curr):
                    err = f"Pattern '{p}' did not match any compartments."
                    raise ValueError(err)
                mask |= curr
        return QuantitySelection(self._ipm, mask)

    def events(self, *patterns: str) -> "QuantitySelection":
        """
        Select events with zero or more pattern strings.

        Specify no patterns to select all events.
        Pattern strings match against event names which combine the source and
        destination compartment names with a separator. e.g., the event
        where individuals transition from `S` to `I` is called `S->I`.
        You must provide both a source and destination pattern, but you can
        use `-`, `>`, or `->` as the separator.

        Multiple patterns are combined as though by boolean-or.
        Pattern strings can use asterisk as a wildcard character
        to match any (non-empty) part of a name besides underscores.
        For example, `"S->*"` would match events `"S->A"` and `"S->B"`.
        `"S->I_*"` would match `"S->I_abc"` and `"S->I_def"`.

        Parameters
        ----------
        *patterns :
            The event selection patterns, as a var-arg.

        Returns
        -------
        :
            The selection object.
        """
        if len(patterns) == 0:
            # select all events
            mask = self._mask(events=True)
        else:
            mask = self._mask()
            for p in patterns:
                src_regex, dst_regex = self._compile_event_pattern(p)
                curr = self._mask(
                    events=[
                        src_regex.fullmatch(src) is not None
                        and dst_regex.fullmatch(dst) is not None
                        for src, dst in (e.tuple for e in self._ipm.events)
                    ]
                )
                if not np.any(curr):
                    err = f"Pattern '{p}' did not match any events."
                    raise ValueError(err)
                mask |= curr
        return QuantitySelection(self._ipm, mask)
