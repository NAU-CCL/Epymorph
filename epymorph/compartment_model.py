"""
The basis of the intra-population model (disease mechanics) system in epymorph.
This represents disease mechanics using a compartmental model for tracking
populations as groupings of integer-numbered individuals.
"""
import re
from dataclasses import dataclass, field
from functools import cached_property
from typing import Iterable, Iterator, Mapping, Sequence

from sympy import Expr, Float, Integer, Symbol

from epymorph.data_shape import DataShapeMatcher, SimDimensions
from epymorph.data_type import DataDType
from epymorph.error import AttributeException, IpmValidationException
from epymorph.simulation import AttributeDef, GeoData, ParamsData
from epymorph.sympy_shim import simplify, simplify_sum, to_symbol
from epymorph.util import (NumpyTypeError, check_ndarray_2, iterator_length,
                           match)

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
            case EdgeDef(_, _, _) as e:
                yield e
            case ForkDef(_, edges):
                for e in edges:
                    yield e


############################################################
# Model Compartments
############################################################


@dataclass(frozen=True)
class CompartmentDef:
    """Defines an IPM compartment."""
    symbol: Symbol
    name: str
    tags: list[str]
    description: str | None = field(default=None)


def compartment(name: str, tags: list[str] | None = None, description: str | None = None) -> CompartmentDef:
    """Define an IPM compartment."""
    if tags is None:
        tags = list()
    return CompartmentDef(to_symbol(name), name, tags, description)


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
class CompartmentSymbols:
    """
    Keeps track of the symbols used in constructing an IPM.
    These symbols are necessary for defining the model's transition rate expressions.
    """
    compartments: list[CompartmentDef]
    attributes: list[AttributeDef]

    def __getitem__(self, name: str) -> Symbol:
        comp = next((c.symbol for c in self.compartments if c.name == name), None)
        if comp is not None:
            return comp
        attr = next((a.symbol for a in self.attributes if a.symbol.name == name), None)
        if attr is not None:
            return attr
        raise KeyError(f"'{name}' does not match a defined IPM symbol.")

    @cached_property
    def compartment_symbols(self) -> list[Symbol]:
        """Accessor for all compartment symbols in definition order."""
        return [c.symbol for c in self.compartments]

    @cached_property
    def attribute_symbols(self) -> list[Symbol]:
        """Accessor for all attribute symbols in definition order."""
        return [a.symbol for a in self.attributes]

    @cached_property
    def all_symbols(self) -> list[Symbol]:
        """Accessor for all model symbols, first compartments then attributes, in definition order."""
        return [*self.compartment_symbols, *self.attribute_symbols]


def create_symbols(compartments: list[CompartmentDef], attributes: list[AttributeDef]) -> CompartmentSymbols:
    """Create a symbols object by combining compartment and attribute definitions."""
    return CompartmentSymbols(compartments, attributes)


@dataclass(frozen=True)
class CompartmentModel:
    """
    A compartment model definition and its corresponding metadata.
    Effectively, a collection of compartments, transitions between compartments,
    and the data parameters which are required to compute the transitions.
    """

    transitions: list[TransitionDef]
    """transition definitions"""
    compartments: list[CompartmentDef]
    """compartment definitions"""
    attributes: list[AttributeDef]
    """attribute definitions"""

    @cached_property
    def num_compartments(self) -> int:
        """The number of compartments in this model."""
        return len(self.compartments)

    @cached_property
    def num_events(self) -> int:
        """The number of distinct events (transitions) in this model."""
        return iterator_length(self.events)

    @cached_property
    def events(self) -> Iterable[EdgeDef]:
        """Iterate over all events in order."""
        return list(_as_events(self.transitions))

    @cached_property
    def compartment_names(self) -> Sequence[str]:
        """The names of all compartments in the order they were declared."""
        return [c.name for c in self.compartments]

    @cached_property
    def compartment_tags(self) -> Sequence[list[str]]:
        """Tags assigned to all IPM compartments."""
        return [c.tags for c in self.compartments]

    @cached_property
    def event_names(self) -> Sequence[str]:
        """The names of all events in the order they were declared."""
        return [f"{e.compartment_from} → {e.compartment_to}"
                for e in self.events]

    @cached_property
    def event_src_dst(self) -> Sequence[tuple[str, str]]:
        """All events represented as a tuple of the source compartment and destination compartment."""
        return [(str(e.compartment_from), str(e.compartment_to)) for e in self.events]

    @cached_property
    def attribute_dtypes(self) -> Mapping[str, DataDType]:
        """The dtypes of all IPM attributes."""
        return {a.name: a.dtype for a in self.attributes}

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

    def compartments_by(self, pattern: str) -> tuple[int, ...]:
        """
        Get the indices of IPM compartments.
        The `pattern` argument supports using asterisk as a wildcard character,
        matching anything besides underscores.
        """
        regex = self._compile_pattern(pattern)
        return tuple((i for i, e in enumerate(self.compartment_names) if regex.match(e)))


def create_model(symbols: CompartmentSymbols, transitions: Iterable[TransitionDef]) -> CompartmentModel:
    """
    Construct a CompartmentModel with the given set of symbols and the given transitions.
    `symbols` must include all of the symbols used in the transition definitions: all compartments and all attributes.
    Raises an IpmValidationException if a valid IPM cannot be constructed from the arguments.
    """
    return _finalize_model(symbols, transitions)


def _finalize_model(symbols: CompartmentSymbols, transitions: Iterable[TransitionDef]) -> CompartmentModel:
    if len(symbols.compartments) == 0:
        msg = "Compartment Model must contain at least one compartment."
        raise IpmValidationException(msg)

    compartment_symbols = [c.symbol for c in symbols.compartments]
    attribute_symbols = [a.symbol for a in symbols.attributes]

    # NOTE: we used to filter out unused compartments and attributes,
    # but that was problematic for the 'no' IPM, and may be counter-intuitive to users.

    trx_comps = _extract_compartments(transitions)
    used_comps = symbols.compartments
    # Filter to just the "in-use" compartments
    # used_comps = [c for c in symbols.compartments
    #              if c.symbol in trx_comps]

    trx_attrs = _extract_symbols(transitions).difference(trx_comps)
    used_attrs = symbols.attributes
    # Filter to just the "in-use" attributes
    # used_attrs = [a for a in symbols.attributes
    #              if a.symbol in trx_attrs]

    # transitions cannot have the source and destination both be exogenous; this would be madness.
    if any((edge.compartment_from in exogenous_states and edge.compartment_to in exogenous_states
            for edge in _as_events(transitions))):
        msg = "Transitions cannot use exogenous states (BIRTH/DEATH) as both source and destination."
        raise IpmValidationException(msg)

    # transitions_compartments minus symbols_compartments should be empty
    missing_comps = trx_comps.difference(compartment_symbols)
    if len(missing_comps) > 0:
        raise IpmValidationException(f"""\
Transitions reference compartments which were not declared as symbols.
Missing states: {", ".join(map(str, missing_comps))}""")

    # transitions_attributes minus symbols_attributes should be empty
    missing_attrs = trx_attrs.difference(attribute_symbols)
    if len(missing_attrs) > 0:
        raise IpmValidationException(f"""\
Transitions reference attributes which were not declared as symbols.
Missing attributes: {", ".join(map(str, missing_attrs))}""")

    return CompartmentModel(list(transitions), used_comps, used_attrs)


def _extract_compartments(trxs: Iterable[TransitionDef]) -> set[Symbol]:
    """Extract the set of compartments used by any transition."""
    return set(compartment
               for e in _as_events(trxs)
               for compartment in [e.compartment_from, e.compartment_to]
               # don't include exogenous states in the compartment set
               if compartment not in exogenous_states)


def _extract_symbols(trxs: Iterable[TransitionDef]) -> set[Symbol]:
    """
    Extract the set of symbols referenced by any transition rate expression.
    This includes compartment symbols.
    """
    return set(symbol
               for e in _as_events(trxs)
               for symbol in e.rate.free_symbols if isinstance(symbol, Symbol))


# Validation


def validate_ipm(ipm: CompartmentModel, dim: SimDimensions, geo_data: GeoData, params_data: ParamsData) -> None:
    """Validate that the IPM has all required attributes."""

    def _validate_attr(attr: AttributeDef) -> Exception | None:
        try:
            name = attr.name
            match attr.source:
                case 'geo':
                    if not name in geo_data:
                        msg = f"Missing geo attribute '{name}'"
                        raise AttributeException(msg)
                    value = geo_data[name]
                case 'params':
                    if not name in params_data:
                        msg = f"Missing params attribute '{name}'"
                        raise AttributeException(msg)
                    value = params_data[name]

            check_ndarray_2(
                value,
                dtype=match.dtype(attr.dtype),
                shape=DataShapeMatcher(attr.shape, dim, True),
            )

            return None
        except NumpyTypeError as e:
            msg = f"Attribute '{attr.name}' is not properly specified. {e}"
            return AttributeException(msg)
        except AttributeException as e:
            return e

    # Collect all attribute errors to raise as a group.
    errors = [x for x in (_validate_attr(attr) for attr in ipm.attributes)
              if x is not None]

    if len(errors) > 0:
        msg = "IPM attribute requirements were not met. See errors:" + \
            "".join(f"\n- {e}" for e in errors)
        raise IpmValidationException(msg)
