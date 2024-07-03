"""
The basis of the intra-population model (disease mechanics) system in epymorph.
This represents disease mechanics using a compartmental model for tracking
populations as groupings of integer-numbered individuals.
"""
import re
from dataclasses import dataclass, field
from functools import cached_property
from typing import Iterable, Iterator, OrderedDict, Sequence

from sympy import Expr, Float, Integer, Symbol

from epymorph.database import AbsoluteName
from epymorph.error import IpmValidationException
from epymorph.simulation import AttributeDef
from epymorph.sympy_shim import simplify, simplify_sum, substitute, to_symbol
from epymorph.util import iterator_length

############################################################
# Model Transitions
############################################################


BIRTH = Symbol('birth_exogenous')
"""An IPM psuedo-compartment representing exogenous input of individuals."""

DEATH = Symbol('death_exogenous')
"""An IPM psuedo-compartment representing exogenous removal of individuals."""

exogenous_states = (BIRTH, DEATH)
"""A complete listing of epymorph-supported exogenous states."""


@dataclass(frozen=True)
class EdgeDef:
    """Defines a single edge transitions in a compartment model."""
    rate: Expr
    compartment_from: Symbol
    compartment_to: Symbol


def edge(compartment_from: Symbol, compartment_to: Symbol, rate: Expr | int | float) -> EdgeDef:
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


def remap_transition(t: TransitionDef, symbol_mapping: dict[Symbol, Symbol]) -> TransitionDef:
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


def compartment(name: str, tags: list[str] | None = None, description: str | None = None) -> CompartmentDef:
    """Define an IPM compartment."""
    return CompartmentDef(name, tags or [], description)


def quick_compartments(symbol_names: str) -> list[CompartmentDef]:
    """
    Define a number of IPM compartments from a space-delimited string.
    This is just short-hand syntax for the `compartment()` function.
    """
    return [compartment(name) for name in symbol_names.split()]


############################################################
# Compartment Symbols
############################################################


@dataclass(frozen=True)
class ModelSymbols:
    """
    Keeps track of the symbols used in constructing an IPM.
    These symbols are necessary for defining the model's transition rate expressions.
    """
    compartments: Sequence[CompartmentDef]
    """The compartments of a model."""
    attributes: OrderedDict[AbsoluteName, AttributeDef]
    """The attributes of a model."""
    compartment_symbols: Sequence[Symbol]
    """Compartment symbols in definition order."""
    attribute_symbols: Sequence[Symbol]
    """Attribute symbols in definition order."""


class CompartmentModel:
    """
    A compartment model definition and its corresponding metadata.
    Effectively, a collection of compartments, transitions between compartments,
    and the data parameters which are required to compute the transitions.
    """

    _symbols: ModelSymbols
    _transitions: list[TransitionDef]

    def __init__(self, symbols: ModelSymbols, transitions: list[TransitionDef]):
        self._symbols = symbols
        self._transitions = transitions
        self._validate()

    @property
    def symbols(self) -> ModelSymbols:
        """The symbols used in the model."""
        return self._symbols

    @property
    def transitions(self) -> Sequence[TransitionDef]:
        """The transitions in the model."""
        return self._transitions

    @property
    def compartments(self) -> Sequence[CompartmentDef]:
        """The compartments in the model."""
        return self.symbols.compartments

    @cached_property
    def num_compartments(self) -> int:
        """The number of compartments in this model."""
        return len(self.symbols.compartments)

    @property
    def attributes(self) -> OrderedDict[AbsoluteName, AttributeDef]:
        """The attributes required by this model."""
        return self.symbols.attributes

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
        return [f"{e.compartment_from} → {e.compartment_to}"
                for e in self.events]

    @cached_property
    def event_src_dst(self) -> Sequence[tuple[str, str]]:
        """All events represented as a tuple of the source compartment and destination compartment."""
        return [(str(e.compartment_from), str(e.compartment_to)) for e in self.events]

    def _compile_pattern(self, pattern: str) -> re.Pattern:
        """Turn a pattern string (which is custom syntax) into a regular expression."""
        # We're not interpreting pattern as a regex directly, so escape any special characters.
        # Then replace '*' with the necessary regex.
        escaped_pattern = re.escape(pattern).replace(r'\*', '[^_]+')
        # Compile with anchors so it matches the entire string.
        return re.compile(f"^{escaped_pattern}$")

    def events_by_src(self, pattern: str) -> tuple[int, ...]:
        """
        Get the indices of IPM events by the source compartment.
        The `pattern` argument supports using asterisk as a wildcard character,
        matching anything besides underscores.
        """
        regex = self._compile_pattern(pattern)
        return tuple((i for i, (src, _) in enumerate(self.event_src_dst) if regex.match(src)))

    def events_by_dst(self, pattern: str) -> tuple[int, ...]:
        """
        Get the indices of IPM events by the destination compartment.
        The `pattern` argument supports using asterisk as a wildcard character,
        matching anything besides underscores.
        """
        regex = self._compile_pattern(pattern)
        return tuple((i for i, (_, dst) in enumerate(self.event_src_dst) if regex.match(dst)))

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
            return next(i for i, (s, d)
                        in enumerate(self.event_src_dst)
                        if s == src and d == dst)
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
        return tuple((i for i, c in enumerate(self.compartments) if regex.match(c.name)))

    def compartment_by_name(self, name: str) -> int:
        """Get a single compartment index by name. Only exact matches are allowed."""
        try:
            return next(i for i, c
                        in enumerate(self.compartments)
                        if c.name == name)
        except StopIteration:
            msg = f"No matching compartment found for name: {name}"
            raise ValueError(msg) from None

    def _validate(self) -> None:
        if len(self.symbols.compartments) == 0:
            msg = "CompartmentModel must contain at least one compartment."
            raise IpmValidationException(msg)

        # Extract the set of compartments used by any transition.
        trx_comps = set(
            compartment
            for e in _as_events(self.transitions)
            for compartment in [e.compartment_from, e.compartment_to]
            # don't include exogenous states in the compartment set
            if compartment not in exogenous_states
        )

        # Extract the set of symbols referenced by any transition rate expression.
        # This includes compartment symbols.
        trx_attrs = set(
            symbol
            for e in _as_events(self.transitions)
            for symbol in e.rate.free_symbols if isinstance(symbol, Symbol)
        ).difference(trx_comps)

        # transitions cannot have the source and destination both be exogenous; this would be madness.
        if any((edge.compartment_from in exogenous_states and edge.compartment_to in exogenous_states
                for edge in _as_events(self.transitions))):
            msg = "Transitions cannot use exogenous states (BIRTH/DEATH) as both source and destination."
            raise IpmValidationException(msg)

        # transitions_compartments minus symbols_compartments should be empty
        missing_comps = trx_comps.difference(self.symbols.compartment_symbols)
        if len(missing_comps) > 0:
            msg = "Transitions reference compartments which were not declared as symbols.\n" \
                f"Missing states: {', '.join(map(str, missing_comps))}"
            raise IpmValidationException(msg)

        # transitions_attributes minus symbols_attributes should be empty
        missing_attrs = trx_attrs.difference(self.symbols.attribute_symbols)
        if len(missing_attrs) > 0:
            msg = "Transitions reference attributes which were not declared as symbols.\n" \
                f"Missing attributes: {', '.join(map(str, missing_attrs))}"
            raise IpmValidationException(msg)


############################################################
# Function-based creation API
############################################################


def create_symbols(compartments: Sequence[CompartmentDef], attributes: Sequence[AttributeDef]) -> ModelSymbols:
    """Create a symbols object by combining compartment and attribute definitions."""
    csym = [to_symbol(c.name) for c in compartments]
    asym = [to_symbol(a.name) for a in attributes]
    return ModelSymbols(
        compartments=list(compartments),
        attributes=OrderedDict([
            (AbsoluteName("gpm:all", "ipm", a.name), a)
            for a in attributes
        ]),
        compartment_symbols=csym,
        attribute_symbols=asym,
    )


def create_model(symbols: ModelSymbols, transitions: Sequence[TransitionDef]) -> CompartmentModel:
    """
    Construct a compartment model with the given set of symbols and the given transitions.
    `symbols` must include all of the symbols used in the transition definitions: all compartments and all attributes.
    Raises an IpmValidationException if a valid IPM cannot be constructed from the arguments.
    """
    return CompartmentModel(symbols, list(transitions))
