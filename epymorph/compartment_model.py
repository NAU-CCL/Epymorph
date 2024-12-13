"""
The basis of the intra-population model (disease mechanics) system in epymorph.
This represents disease mechanics using a compartmental model for tracking
populations as groupings of integer-numbered individuals.
"""

import dataclasses
import re
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
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
)

import numpy as np
from numpy.typing import NDArray
from sympy import Add, Expr, Float, Integer, Symbol
from typing_extensions import override

from epymorph.database import AbsoluteName, AttributeDef
from epymorph.error import IpmValidationException
from epymorph.simulation import DEFAULT_STRATA, META_STRATA, gpm_strata
from epymorph.sympy_shim import simplify, simplify_sum, substitute, to_symbol
from epymorph.util import (
    acceptable_name,
    are_instances,
    are_unique,
    filter_unique,
    iterator_length,
)

######################
# Model Compartments #
######################

BIRTH = Symbol("birth_exogenous")
"""An IPM psuedo-compartment representing exogenous input of individuals."""

DEATH = Symbol("death_exogenous")
"""An IPM psuedo-compartment representing exogenous removal of individuals."""

exogenous_states = (BIRTH, DEATH)
"""A complete listing of epymorph-supported exogenous states."""


@dataclass(frozen=True)
class CompartmentName:
    base: str
    subscript: str | None
    strata: str | None
    full: str = field(init=False, hash=False, compare=False)

    def __post_init__(self):
        full = "_".join(
            x for x in (self.base, self.subscript, self.strata) if x is not None
        )
        if acceptable_name.match(full) is None:
            raise ValueError(f"Invalid compartment name: {full}")
        object.__setattr__(self, "full", full)

    def with_subscript(self, subscript: str | None) -> Self:
        if self.subscript == "exogenous":
            return self
        return dataclasses.replace(self, subscript=subscript)

    def with_strata(self, strata: str | None) -> Self:
        if self.subscript == "exogenous":
            return self
        return dataclasses.replace(self, strata=strata)

    def __str__(self) -> str:
        return self.full

    @classmethod
    def parse(cls, name: str) -> Self:
        if (i := name.find("_")) != -1:
            return cls(name[0:i], name[i + 1 :], None)
        return cls(name, None, None)


@dataclass(frozen=True)
class CompartmentDef:
    """Defines an IPM compartment."""

    name: CompartmentName
    tags: list[str]
    description: str | None = field(default=None)

    def with_strata(self, strata: str) -> Self:
        return dataclasses.replace(self, name=self.name.with_strata(strata))


def compartment(
    name: str,
    tags: list[str] | None = None,
    description: str | None = None,
) -> CompartmentDef:
    """Define an IPM compartment."""
    return CompartmentDef(CompartmentName.parse(name), tags or [], description)


def quick_compartments(symbol_names: str) -> list[CompartmentDef]:
    """
    Define a number of IPM compartments from a space-delimited string.
    This is just short-hand syntax for the `compartment()` function.
    """
    return [compartment(name) for name in symbol_names.split()]


#####################
# Model Transitions #
#####################


@dataclass(frozen=True)
class EdgeName:
    compartment_from: CompartmentName
    compartment_to: CompartmentName
    full: str = field(init=False, hash=False, compare=False)

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
        return dataclasses.replace(
            self,
            compartment_from=self.compartment_from.with_subscript(subscript),
            compartment_to=self.compartment_to.with_subscript(subscript),
        )

    def with_strata(self, strata: str | None) -> Self:
        return dataclasses.replace(
            self,
            compartment_from=self.compartment_from.with_strata(strata),
            compartment_to=self.compartment_to.with_strata(strata),
        )

    def __str__(self) -> str:
        return self.full


@dataclass(frozen=True)
class EdgeDef:
    """Defines a single edge transitions in a compartment model."""

    name: EdgeName
    rate: Expr
    compartment_from: Symbol
    compartment_to: Symbol

    @property
    def tuple(self) -> tuple[str, str]:
        return str(self.compartment_from), str(self.compartment_to)


def edge(
    compartment_from: Symbol,
    compartment_to: Symbol,
    rate: Expr | int | float,
) -> EdgeDef:
    """Define a transition edge from one compartment to another at the given rate."""
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
    """Defines a fork-style transition in a compartment model."""

    rate: Expr
    edges: list[EdgeDef]
    probs: list[Expr]

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

    For example, consider two edges given rates:
    1. `delta * EXPOSED * rho`
    2. `delta * EXPOSED * (1 - rho)`

    `delta * EXPOSED` is the base rate and `rho` describes the proportional split for
    each edge.
    """

    # First verify that the edges all come from the same state.
    if len(set(e.compartment_from for e in edges)) > 1:
        msg = (
            "In a Fork, all edges must share the same `state_from`.\n"
            f"  Problem in: {str(edges)}"
        )
        raise IpmValidationException(msg)
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
"""All ways to define a compartment model transition: edges or forks."""


def _as_events(trxs: Iterable[TransitionDef]) -> Iterator[EdgeDef]:
    """
    Iterator for all unique events defined in the transition model.
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
    """
    Replaces symbols used in the transition using substitution from `symbol_mapping`.
    """
    match t:
        case EdgeDef():
            return _remap_edge(t, strata, symbol_mapping)
        case ForkDef():
            return _remap_fork(t, strata, symbol_mapping)


######################
# Compartment Models #
######################


class ModelSymbols:
    """IPM symbols needed in defining the model's transition rate expressions."""

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
        """Select compartment symbols by name."""
        return [self._csymbols[n] for n in names]

    def requirements(self, *names: str) -> Sequence[Symbol]:
        """Select requirement symbols by name."""
        return [self._rsymbols[n] for n in names]


class BaseCompartmentModel(ABC):
    """Shared base-class for compartment models."""

    _abstract_model = True  # marking this abstract skips metaclass validation

    compartments: Sequence[CompartmentDef] = ()
    """The compartments of the model."""

    requirements: Sequence[AttributeDef] = ()
    """The attributes required by the model."""

    # NOTE: these two attributes are coded as such so that overriding
    # this class is simpler for users. Normally I'd make them properties,
    # -- since they really should not be modified after creation --
    # but this would increase the implementation complexity.
    # And to avoid requiring users to call the initializer, the rest
    # of the attributes are cached_properties which initialize lazily.

    @cached_property
    @abstractmethod
    def symbols(self) -> ModelSymbols:
        """The symbols which represent parts of this model."""

    @cached_property
    @abstractmethod
    def transitions(self) -> Sequence[TransitionDef]:
        """The transitions in the model."""

    @property
    def quantities(self) -> Iterator[CompartmentDef | EdgeDef]:
        yield from self.compartments
        yield from self.events

    @cached_property
    @abstractmethod
    def requirements_dict(self) -> OrderedDict[AbsoluteName, AttributeDef]:
        """The attributes required by this model."""

    @cached_property
    def num_compartments(self) -> int:
        """The number of compartments in this model."""
        return len(self.compartments)

    @cached_property
    def events(self) -> Sequence[EdgeDef]:
        """Iterate over all events in order."""
        return list(_as_events(self.transitions))

    @cached_property
    def num_events(self) -> int:
        """The number of distinct events (transitions) in this model."""
        return iterator_length(self.events)

    @property
    @abstractmethod
    def strata(self) -> Sequence[str]:
        """The names of the strata involved in this compartment model."""

    @property
    @abstractmethod
    def is_multistrata(self) -> bool:
        """True if this compartment model is multistrata (False for single-strata)."""

    @property
    def select(self) -> "QuantitySelector":
        return QuantitySelector(self)


####################################
# Single-strata Compartment Models #
####################################


class CompartmentModelClass(ABCMeta):
    """
    The metaclass for user-defined CompartmentModel classes.
    Used to verify proper class implementation.
    """

    def __new__(
        mcs: Type["CompartmentModelClass"],
        name: str,
        bases: tuple[type, ...],
        dct: dict[str, Any],
    ) -> "CompartmentModelClass":
        # Skip these checks for classes we want to treat as abstract:
        if dct.get("_abstract_model", False):
            return super().__new__(mcs, name, bases, dct)

        # Check model compartments.
        cmps = dct.get("compartments")
        if cmps is None or not isinstance(cmps, (list, tuple)):
            raise TypeError(
                f"Invalid compartments in {name}: please specify as a list or tuple."
            )
        if len(cmps) == 0:
            raise TypeError(
                f"Invalid compartments in {name}: "
                "please specify at least one compartment."
            )
        if not are_instances(cmps, CompartmentDef):
            raise TypeError(
                f"Invalid compartments in {name}: must be instances of CompartmentDef."
            )
        if not are_unique(c.name for c in cmps):
            raise TypeError(
                f"Invalid compartments in {name}: compartment names must be unique."
            )
        # Make compartments immutable.
        dct["compartments"] = tuple(cmps)

        # Check transitions... we have to instantiate the class.
        cls = super().__new__(mcs, name, bases, dct)
        instance = cls()

        trxs = instance.transitions

        # transitions cannot have the source and destination both be exogenous;
        # this would be madness.
        if any(
            edge.compartment_from in exogenous_states
            and edge.compartment_to in exogenous_states
            for edge in _as_events(trxs)
        ):
            raise TypeError(
                f"Invalid transitions in {name}: "
                "transitions cannot use exogenous states (BIRTH/DEATH) "
                "as both source and destination."
            )

        # Extract the set of compartments used by transitions.
        trx_comps = set(
            compartment
            for e in _as_events(trxs)
            for compartment in [e.compartment_from, e.compartment_to]
            # don't include exogenous states in the compartment set
            if compartment not in exogenous_states
        )

        # Extract the set of requirements used by transition rate expressions
        # by taking all used symbols and subtracting compartment symbols.
        trx_reqs = set(
            symbol
            for e in _as_events(trxs)
            for symbol in e.rate.free_symbols
            if isinstance(symbol, Symbol)
        ).difference(trx_comps)

        # transition compartments minus declared compartments should be empty
        missing_comps = trx_comps.difference(instance.symbols.all_compartments)
        if len(missing_comps) > 0:
            raise TypeError(
                f"Invalid transitions in {name}: "
                "transitions reference compartments which were not declared.\n"
                f"Missing compartments: {', '.join(map(str, missing_comps))}"
            )

        # transition requirements minus declared requirements should be empty
        missing_reqs = trx_reqs.difference(instance.symbols.all_requirements)
        if len(missing_reqs) > 0:
            raise TypeError(
                f"Invalid transitions in {name}: "
                "transitions reference requirements which were not declared.\n"
                f"Missing requirements: {', '.join(map(str, missing_reqs))}"
            )

        return cls


class CompartmentModel(BaseCompartmentModel, ABC, metaclass=CompartmentModelClass):
    """
    A compartment model definition and its corresponding metadata.
    Effectively, a collection of compartments, transitions between compartments,
    and the data parameters which are required to compute the transitions.
    """

    _abstract_model = True  # marking this abstract skips metaclass validation

    @cached_property
    def symbols(self) -> ModelSymbols:
        """The symbols which represent parts of this model."""
        return ModelSymbols(
            [(c.name.full, c.name.full) for c in self.compartments],
            [(r.name, r.name) for r in self.requirements],
        )

    @cached_property
    def requirements_dict(self) -> OrderedDict[AbsoluteName, AttributeDef]:
        """The attributes required by this model."""
        return OrderedDict(
            [
                (AbsoluteName(gpm_strata(DEFAULT_STRATA), "ipm", r.name), r)
                for r in self.requirements
            ]
        )

    @cached_property
    def transitions(self) -> Sequence[TransitionDef]:
        """The transitions in the model."""
        return self.edges(self.symbols)

    @abstractmethod
    def edges(self, symbols: ModelSymbols) -> Sequence[TransitionDef]:
        """
        When implementing a CompartmentModel, override this method
        to build the transition edges between compartments. You are
        given a reference to this model's symbols library so you can
        build expressions for the transition rates.
        """

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


class MultistrataModelSymbols(ModelSymbols):
    """IPM symbols needed in defining the model's transition rate expressions."""

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
        """
        sym = self._strata_symbols[strata]
        return sym.all_compartments if len(names) == 0 else sym.compartments(*names)

    def strata_requirements(self, strata: str, *names: str) -> Sequence[Symbol]:
        """
        Select requirement symbols by name in a particular strata.
        If `names` is non-empty, select those symbols by their original name.
        If `names` is empty, return all symbols.
        """
        sym = self._strata_symbols[strata]
        return sym.all_requirements if len(names) == 0 else sym.requirements(*names)


MetaEdgeBuilder = Callable[[MultistrataModelSymbols], Sequence[TransitionDef]]
"""A function for creating meta edges in a multistrata RUME."""


class CombinedCompartmentModel(BaseCompartmentModel):
    """A CompartmentModel constructed by combining others."""

    compartments: Sequence[CompartmentDef]
    """All compartments; renamed with strata."""
    requirements: Sequence[AttributeDef]
    """All requirements, including meta-requirements."""

    _strata: Sequence[tuple[str, CompartmentModel]]
    _meta_requirements: Sequence[AttributeDef]
    _meta_edges: MetaEdgeBuilder

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

    @cached_property
    def symbols(self) -> MultistrataModelSymbols:
        """The symbols which represent parts of this model."""
        return MultistrataModelSymbols(
            strata=self._strata, meta_requirements=self._meta_requirements
        )

    @cached_property
    def transitions(self) -> Sequence[TransitionDef]:
        symbols = self.symbols

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

        return [
            *(
                _remap_transition(trx, strata, mapping)
                for (strata, ipm), mapping in zip(self._strata, strata_mapping)
                for trx in ipm.transitions
            ),
            *(fix_edge_names(x) for x in self._meta_edges(symbols)),
        ]

    @cached_property
    def requirements_dict(self) -> OrderedDict[AbsoluteName, AttributeDef]:
        return OrderedDict(
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


class QuantityGroupResult(NamedTuple):
    """The result of a quantity grouping operation."""

    groups: tuple[Quantity, ...]
    """The quantities (or psuedo-quantities) representing each group."""
    indices: tuple[tuple[int, ...], ...]
    """The IPM quantity indices included in each group."""


_N = TypeVar("_N", bound=CompartmentName | EdgeName)


class QuantityGrouping(NamedTuple):
    """Describes how to group simulation output quantities (events and compartments).
    The default combines any quantity whose names match exactly. This is common in
    multistrata models where events from several strata impact one transition.
    You can also choose to group across strata and subscript differences.
    Setting `strata` or `subscript` to True means those elements of quantity names
    (if they exist) are ignored for the purposes of matching."""

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


@dataclass(frozen=True)
class QuantityStrategy:
    """A strategy for dealing with the quantity axis, e.g., in processing results.
    Quantities here are an IPM's compartments and events.

    Strategies can include selection of a subset, grouping, and aggregation."""

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
        """The quantities in the result. If the strategy performs grouping these
        may be pseudo-quantities made by combining the quantities in the group."""

    @property
    @abstractmethod
    def labels(self) -> Sequence[str]:
        """Labels for the quantities in the result, after any grouping."""

    def disambiguate(self) -> OrderedDict[str, str]:
        """Creates a name mapping to disambiguate IPM quantities that have
        the same name. This happens commonly in multistrata IPMs with
        meta edges where multiple other strata influence a transmission rate
        in a single strata. The returned mapping includes only the selected IPM
        compartments and events, but definition order is maintained.
        Keys are the unique name and values are the original names
        (because the original names might contain duplicates);
        so you will have to map into unique names by position, but can map
        back using this mapping directly."""
        selected = [
            (i, q) for i, q in enumerate(self.ipm.quantities) if self.selection[i]
        ]
        qs_original = [q.name.full for i, q in selected]
        qs_renamed = [f"{q.name}_{i}" for i, q in selected]
        return OrderedDict(zip(qs_renamed, qs_original))

    def disambiguate_groups(self) -> OrderedDict[str, str]:
        """Like method `disambiguate()` but for working with quantities
        after any grouping has been performed. If grouping is None,
        this is equivalent to `disambiguate()`."""
        if self.grouping is None:
            return self.disambiguate()
        groups, _ = self.grouping.map(self.selected)
        selected = [(i, q) for i, q in enumerate(groups)]
        qs_original = [q.name.full for i, q in selected]
        qs_renamed = [f"{q.name}_{i}" for i, q in selected]
        return OrderedDict(zip(qs_renamed, qs_original))


@dataclass(frozen=True)
class QuantitySelection(QuantityStrategy):
    """Describe a sub-selection of IPM quantities, which are its
    events and compartments (no grouping or aggregation)."""

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
        """Return the selected compartment index, if and only if there is exactly
        one compartment in the selection. Otherwise, raises ValueError
        See `compartment_indices()` if you want to select possibly multiple indices."""
        indices = self.compartment_indices
        if len(indices) != 1:
            err = (
                "Your selection must contain exactly one compartment to use this "
                "method. Use `compartment_indices()` if you want to select more."
            )
            raise ValueError(err)
        return indices[0]

    @property
    def compartment_indices(self) -> tuple[int, ...]:
        """Return the selected compartment indices. These indices may be useful
        for instance to access a simulation output's `compartments` result array.
        May be an empty tuple."""
        C = self.ipm.num_compartments
        return tuple(i for i in np.flatnonzero(self.selection) if i < C)

    @property
    def event_index(self) -> int:
        """Return the selected event index, if and only if there is exactly
        one event in the selection. Otherwise, raises ValueError.
        See `event_indices()` if you want to select possibly multiple indices."""
        indices = self.event_indices
        if len(indices) != 1:
            err = (
                "Your selection must contain exactly one event to use this "
                "method. Use `event_indices()` if you want to select more."
            )
            raise ValueError(err)
        return indices[0]

    @property
    def event_indices(self) -> tuple[int, ...]:
        """Return the selected event indices. These indices may be useful
        for instance to access a simulation output's `events` result array.
        May be an empty tuple."""
        C = self.ipm.num_compartments
        return tuple(i - C for i in np.flatnonzero(self.selection) if i >= C)

    def group(
        self,
        *,
        strata: bool = False,
        subscript: bool = False,
    ) -> "QuantityGroup":
        """Groups quantities according to the given options.

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
        """
        return QuantityGroup(
            self.ipm,
            self.selection,
            QuantityGrouping(strata, subscript),
        )


@dataclass(frozen=True)
class QuantityGroup(QuantityStrategy):
    """Describes a group operation on IPM quantities,
    with an optional sub-selection."""

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

    def sum(self) -> "QuantityAggregation":
        """Combine grouped quantities by adding their values."""
        return QuantityAggregation(self.ipm, self.selection, self.grouping, "sum")


@dataclass(frozen=True)
class QuantityAggregation(QuantityStrategy):
    """Describes a group-and-aggregate operation on IPM quantities,
    with an optional sub-selection."""

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
    """A utility class for selecting a subset of IPM quantities."""

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
        """Select all compartments and events."""
        m = self._mask()
        m[:] = True
        return QuantitySelection(self._ipm, m)

    def indices(self, *indices: int) -> "QuantitySelection":
        """Select quantities by index (determined by IPM definition order:
        all IPM compartments, all IPM events, and then meta edge events if any)."""
        m = self._mask()
        m[indices] = True
        return QuantitySelection(self._ipm, m)

    def _compile_pattern(self, pattern: str) -> re.Pattern:
        """Turn a pattern string (which is custom syntax) into a regular expression."""
        # We're not interpreting pattern as a regex directly, so escape any
        # special characters. Then replace '*' with the necessary regex.
        return re.compile(re.escape(pattern).replace(r"\*", ".*?"))

    def _compile_event_pattern(self, pattern: str) -> tuple[re.Pattern, re.Pattern]:
        """Interpret a pattern string as two patterns matching against the
        source and destination compartments."""
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
        """Select compartments and events by providing pattern strings for each.

        Providing an empty sequence implies selecting none of that type.
        Multiple patterns are combined as though by boolean-or.
        """
        cs = [compartments] if isinstance(compartments, str) else [*compartments]
        es = [events] if isinstance(events, str) else [*events]
        c_mask = self._mask() if len(cs) == 0 else self.compartments(*cs).selection
        e_mask = self._mask() if len(es) == 0 else self.events(*es).selection
        return QuantitySelection(self._ipm, c_mask | e_mask)

    def compartments(self, *patterns: str) -> "QuantitySelection":
        """Select compartments with zero or more pattern strings.

        Specify no patterns to select all compartments.
        Pattern strings match against compartment names.
        Multiple patterns are combined as though by boolean-or.
        Pattern strings can use asterisk as a wildcard character
        to match any (non-empty) part of a name besides underscores.
        For example, "I_*" would match events "I_abc" and "I_def".
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
        """Select events with zero or more pattern strings.

        Specify no patterns to select all events.
        Pattern strings match against event names which combine the source and
        destination compartment names with a separator. e.g., the event
        where individuals transition from "S" to "I" is called "S->I".
        You must provide both a source and destination pattern, but you can
        use "-", ">", or "->" as the separator.
        Multiple patterns are combined as though by boolean-or.
        Pattern strings can use asterisk as a wildcard character
        to match any (non-empty) part of a name besides underscores.
        For example, "S->*" would match events "S->A" and "S->B".
        "S->I_*" would match "S->I_abc" and "S->I_def".
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
