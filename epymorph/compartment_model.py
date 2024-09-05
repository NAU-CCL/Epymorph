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
from typing import Any, Callable, Iterable, Iterator, OrderedDict, Sequence, Type

from sympy import Expr, Float, Integer, Symbol

from epymorph.database import AbsoluteName
from epymorph.error import IpmValidationException
from epymorph.simulation import DEFAULT_STRATA, META_STRATA, AttributeDef, gpm_strata
from epymorph.sympy_shim import simplify, simplify_sum, substitute, to_symbol
from epymorph.util import acceptable_name, are_instances, are_unique, iterator_length

############################################################
# Model Transitions
############################################################


BIRTH = Symbol("birth_exogenous")
"""An IPM psuedo-compartment representing exogenous input of individuals."""

DEATH = Symbol("death_exogenous")
"""An IPM psuedo-compartment representing exogenous removal of individuals."""

exogenous_states = (BIRTH, DEATH)
"""A complete listing of epymorph-supported exogenous states."""


@dataclass(frozen=True)
class EdgeDef:
    """Defines a single edge transitions in a compartment model."""

    rate: Expr
    compartment_from: Symbol
    compartment_to: Symbol


def edge(
    compartment_from: Symbol, compartment_to: Symbol, rate: Expr | int | float
) -> EdgeDef:
    """Define a transition edge going from one compartment to another at the given rate."""
    if isinstance(rate, int):
        _rate = Integer(rate)
    elif isinstance(rate, float):
        _rate = Float(rate)
    else:
        _rate = rate
    return EdgeDef(_rate, compartment_from, compartment_to)


@dataclass(frozen=True)
class ForkDef:
    """Defines a fork-style transition in a compartment model."""

    rate: Expr
    edges: list[EdgeDef]
    probs: list[Expr]


def fork(*edges: EdgeDef) -> ForkDef:
    """
    Define a forked transition: a set of edges that come from the same compartment but go to different compartments.
    It is assumed the edges will share a "base rate" -- a common sub-expression among all edge rates --
    and that each edge in the fork is given a proportion on that base rate.

    For example, consider two edges given rates:
    1. `delta * EXPOSED * rho`
    2. `delta * EXPOSED * (1 - rho)`

    `delta * EXPOSED` is the base rate and `rho` describes the proportional split for each edge.
    """

    # First verify that the edges all come from the same state.
    if len(set(e.compartment_from for e in edges)) > 1:
        msg = f"In a Fork, all edges must share the same `state_from`.\n  Problem in: {str(edges)}"
        raise IpmValidationException(msg)
    # it is assumed the fork's edges are defined with complementary rate expressions, e.g.,
    edge_rates = [e.rate for e in edges]
    # the "base rate" -- how many individuals transition on any of these edges --
    # is the sum of all the edge rates (this defines the lambda for the poisson draw)
    rate = simplify_sum(edge_rates)
    # the probability of following a particular edge is then the edge's rate divided by the base rate
    # (this defines the probability split in the eventual multinomial draw)
    probs = [simplify(r / rate) for r in edge_rates]  # type: ignore
    return ForkDef(rate, list(edges), probs)


TransitionDef = EdgeDef | ForkDef
"""All ways to define a compartment model transition: edges or forks."""


def _as_events(trxs: Iterable[TransitionDef]) -> Iterator[EdgeDef]:
    """
    Iterator for all unique events defined in the transition model.
    Each edge corresponds to a single event, even the edges that are part of a fork.
    The events are returned in a stable order (definition order) so that they can be indexed that way.
    """
    for t in trxs:
        match t:
            case EdgeDef() as e:
                yield e
            case ForkDef(_, edges):
                for e in edges:
                    yield e


def _remap_edge(e: EdgeDef, symbol_mapping: dict[Symbol, Symbol]) -> EdgeDef:
    return EdgeDef(
        rate=substitute(e.rate, symbol_mapping),
        compartment_from=symbol_mapping[e.compartment_from],
        compartment_to=symbol_mapping[e.compartment_to],
    )


def _remap_fork(f: ForkDef, symbol_mapping: dict[Symbol, Symbol]) -> ForkDef:
    return ForkDef(
        rate=substitute(f.rate, symbol_mapping),
        edges=[_remap_edge(e, symbol_mapping) for e in f.edges],
        probs=[substitute(p, symbol_mapping) for p in f.probs],
    )


def _remap_transition(
    t: TransitionDef, symbol_mapping: dict[Symbol, Symbol]
) -> TransitionDef:
    """Replaces all symbols used in the transition using substitution from `symbol_mapping`."""
    match t:
        case EdgeDef():
            return _remap_edge(t, symbol_mapping)
        case ForkDef():
            return _remap_fork(t, symbol_mapping)


############################################################
# Model Compartments
############################################################


@dataclass(frozen=True)
class CompartmentDef:
    """Defines an IPM compartment."""

    name: str
    tags: list[str]
    description: str | None = field(default=None)

    def __post_init__(self):
        if acceptable_name.match(self.name) is None:
            raise ValueError(f"Invalid compartment name: {self.name}")


def compartment(
    name: str, tags: list[str] | None = None, description: str | None = None
) -> CompartmentDef:
    """Define an IPM compartment."""
    return CompartmentDef(name, tags or [], description)


def quick_compartments(symbol_names: str) -> list[CompartmentDef]:
    """
    Define a number of IPM compartments from a space-delimited string.
    This is just short-hand syntax for the `compartment()` function.
    """
    return [compartment(name) for name in symbol_names.split()]


############################################################
# Compartment Models
############################################################


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
        # NOTE: the arguments here are tuples of name and symbolic name; this is redundant for
        # single-strata models, but allows multistrata models to keep fine-grained control over
        # symbol substitution while allowing the user to refer to the names they already know.
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

    @cached_property
    def event_names(self) -> Sequence[str]:
        """The names of all events in the order they were declared."""
        return [f"{e.compartment_from} → {e.compartment_to}" for e in self.events]

    @cached_property
    def event_src_dst(self) -> Sequence[tuple[str, str]]:
        """All events represented as a tuple of the source compartment and destination compartment."""
        return [(str(e.compartment_from), str(e.compartment_to)) for e in self.events]

    def _compile_pattern(self, pattern: str) -> re.Pattern:
        """Turn a pattern string (which is custom syntax) into a regular expression."""
        # We're not interpreting pattern as a regex directly, so escape any special characters.
        # Then replace '*' with the necessary regex.
        escaped_pattern = re.escape(pattern).replace(r"\*", "[^_]+")
        # Compile with anchors so it matches the entire string.
        return re.compile(f"^{escaped_pattern}$")

    def events_by_src(self, pattern: str) -> tuple[int, ...]:
        """
        Get the indices of IPM events by the source compartment.
        The `pattern` argument supports using asterisk as a wildcard character,
        matching anything besides underscores.
        """
        regex = self._compile_pattern(pattern)
        return tuple(
            (i for i, (src, _) in enumerate(self.event_src_dst) if regex.match(src))
        )

    def events_by_dst(self, pattern: str) -> tuple[int, ...]:
        """
        Get the indices of IPM events by the destination compartment.
        The `pattern` argument supports using asterisk as a wildcard character,
        matching anything besides underscores.
        """
        regex = self._compile_pattern(pattern)
        return tuple(
            (i for i, (_, dst) in enumerate(self.event_src_dst) if regex.match(dst))
        )

    def event_by_name(self, name: str) -> int:
        """Get a single event index by name. For example: "S->I". Only exact matches are allowed."""
        try:
            if "->" in name:
                src, dst = name.split("->")
            elif "-" in name:
                src, dst = name.split("-")
            elif ">" in name:
                src, dst = name.split(">")
            else:
                raise ValueError(f"Invalid event name syntax: {name}")
        except ValueError:
            raise ValueError(f"Invalid event name syntax: {name}") from None

        try:
            return next(
                i
                for i, (s, d) in enumerate(self.event_src_dst)
                if s == src and d == dst
            )
        except StopIteration:
            msg = f"No matching event found for name: {name}"
            raise ValueError(msg) from None

    def compartments_by(self, pattern: str) -> tuple[int, ...]:
        """
        Get the indices of IPM compartments.
        The `pattern` argument supports using asterisk as a wildcard character,
        matching anything besides underscores.
        """
        regex = self._compile_pattern(pattern)
        return tuple(
            (i for i, c in enumerate(self.compartments) if regex.match(c.name))
        )

    def compartment_by_name(self, name: str) -> int:
        """Get a single compartment index by name. Only exact matches are allowed."""
        try:
            return next(i for i, c in enumerate(self.compartments) if c.name == name)
        except StopIteration:
            msg = f"No matching compartment found for name: {name}"
            raise ValueError(msg) from None


############################################################
# Single-strata Compartment Models
############################################################


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
                f"Invalid compartments in {name}: please specify at least one compartment."
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

        # transitions cannot have the source and destination both be exogenous; this would be madness.
        if any(
            edge.compartment_from in exogenous_states
            and edge.compartment_to in exogenous_states
            for edge in _as_events(trxs)
        ):
            raise TypeError(
                f"Invalid transitions in {name}: "
                "transitions cannot use exogenous states (BIRTH/DEATH) as both source and destination."
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
            [(c.name, c.name) for c in self.compartments],
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


############################################################
# Multi-strata Compartment Models
############################################################


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
            (c.name, strata_name, f"{c.name}_{strata_name}")
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
            dataclasses.replace(comp, name=f"{comp.name}_{strata_name}")
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
        all_cs = iter(symbols.all_compartments)
        all_rs = iter(symbols.all_requirements)
        for _, ipm in self._strata:
            mapping = {}
            old = ipm.symbols
            for old_symbol in old.all_compartments:
                mapping[old_symbol] = next(all_cs)
            for old_symbol in old.all_requirements:
                mapping[old_symbol] = next(all_rs)
            strata_mapping.append(mapping)

        return [
            *(
                _remap_transition(trx, mapping)
                for (_, ipm), mapping in zip(self._strata, strata_mapping)
                for trx in ipm.transitions
            ),
            *self._meta_edges(symbols),
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
