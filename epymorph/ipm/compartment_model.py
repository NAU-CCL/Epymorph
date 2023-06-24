from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Generator, Iterable, cast

import numpy as np
import sympy
from numpy.typing import NDArray
from sympy import Expr, Symbol

from epymorph.ipm.attribute import AttributeDef
from epymorph.util import index_where, iterator_length


def _to_symbol(name: str) -> Symbol:
    # sympy doesn't provide explicit typing, so we adapt
    return cast(Symbol, sympy.symbols(name))


class InvalidModelException(Exception):
    pass

############################################################
# Model Transitions
############################################################


@dataclass(frozen=True)
class Transition(ABC):
    rate: Expr

    @staticmethod
    def as_events(trxs: Iterable[TransitionDef]) -> Generator[tuple[(int, EdgeDef)], None, None]:
        """
        Iterator for all unique events defined in the transition model.
        Each edge corresponds to a single event, even the edges that are part of a fork.
        The events are returned in a stable order (definition order) so that they can be indexed that way.
        """
        index = 0
        for t in trxs:
            match t:
                case EdgeDef(_, _, _) as e:
                    yield (index, e)
                    index += 1
                case ForkDef(_, edges):
                    for e in edges:
                        yield (index, e)
                        index += 1

    @staticmethod
    def event_count(trx: Iterable[TransitionDef]) -> int:
        """Count the number of unique events in this transition model."""
        return iterator_length(Transition.as_events(trx))

    @staticmethod
    def extract_compartments(trxs: Iterable[TransitionDef]) -> set[Symbol]:
        """Extract the set of compartments used by any transition."""
        return set(compartment
                   for _, e in Transition.as_events(trxs)
                   for compartment in [e.compartment_from, e.compartment_to])

    @staticmethod
    def extract_symbols(trxs: Iterable[TransitionDef]) -> set[Symbol]:
        """
        Extract the set of symbols referenced by any transition rate expression.
        This includes compartment symbols.
        """
        return set(symbol
                   for _, e in Transition.as_events(trxs)
                   for symbol in e.rate.free_symbols if isinstance(symbol, Symbol))


@dataclass(frozen=True)
class EdgeDef(Transition):
    compartment_from: Symbol
    compartment_to: Symbol


def edge(compartment_from: Symbol, compartment_to: Symbol, rate: Expr) -> EdgeDef:
    """Define a transition edge going from one compartment to another at the given rate."""
    return EdgeDef(rate, compartment_from, compartment_to)


@dataclass(frozen=True)
class ForkDef(Transition):
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
        raise InvalidModelException(
            f"In a Fork, all edges must share the same `state_from`.\n  Problem in: {str(edges)}")
    # it is assumed the fork's edges are defined with complementary rate expressions, e.g.,
    edge_rates = [e.rate for e in edges]
    # the "base rate" -- how many individuals transition on any of these edges --
    # is the sum of all the edge rates (this defines the lambda for the poisson draw)
    rate = sympy.simplify(sympy.Add(*edge_rates))
    # the probability of following a particular edge is then the edge's rate divided by the base rate
    # (this defines the probability split in the eventual multinomial draw)
    probs = [sympy.simplify(r / rate) for r in edge_rates]
    return ForkDef(rate, list(edges), probs)


TransitionDef = EdgeDef | ForkDef

############################################################
# Model Compartments
############################################################


@dataclass(frozen=True)
class CompartmentDef:
    symbol: Symbol
    name: str


def compartment(symbol_name: str, name: str | None = None) -> CompartmentDef:
    if name is None:
        name = symbol_name
    return CompartmentDef(_to_symbol(symbol_name), name)


def quick_compartments(symbol_names: str) -> list[CompartmentDef]:
    return [compartment(name) for name in symbol_names.split()]


############################################################
# Compartment Symbols
############################################################


@dataclass(frozen=True)
class CompartmentSymbols:
    compartments: list[CompartmentDef]
    attributes: list[AttributeDef]
    compartment_symbols: list[Symbol]
    attribute_symbols: list[Symbol]
    all_symbols: list[Symbol]


def create_symbols(compartments: list[CompartmentDef], attributes: list[AttributeDef]) -> CompartmentSymbols:
    compartment_symbols = [c.symbol for c in compartments]
    attribute_symbols = [a.symbol for a in attributes]
    return CompartmentSymbols(
        compartments,
        attributes,
        compartment_symbols,
        attribute_symbols,
        compartment_symbols + attribute_symbols)


@dataclass(frozen=True)
class CompartmentModel:
    transitions: list[TransitionDef]
    compartments: list[CompartmentDef]
    attributes: list[AttributeDef]
    # a matrix defining how each event impacts each compartment (subtracting or adding individuals)
    apply_matrix: NDArray[np.int_]
    # mapping from compartment index to the list of event indices which source from that compartment
    events_leaving_compartment: list[list[int]]
    # mapping from event index to the compartment index it sources from
    source_compartment_for_event: list[int]


def create_model(symbols: CompartmentSymbols, transitions: list[TransitionDef]) -> CompartmentModel:
    # Our main task is to verify that the transitions specified are aligned with the symbols provided,
    # then pre-compute some useful metadata about the model.

    # Make sure all transition compartments and attributes are defined in symbols.
    trx_comps = Transition.extract_compartments(transitions)
    trx_attrs = Transition.extract_symbols(transitions)\
        .difference(trx_comps)  # attributes are symbols that are not compartments

    # transitions_compartments minus symbols_compartments should be empty
    missing_comps = trx_comps.difference(symbols.compartment_symbols)
    if len(missing_comps) > 0:
        raise InvalidModelException(f"""\
Transitions reference compartments which were not declared as symbols.
Missing states: {", ".join(map(str, missing_comps))}""")

    # transitions_attributes minus symbols_attributes should be empty
    missing_attrs = trx_attrs.difference(symbols.attribute_symbols)
    if len(missing_attrs) > 0:
        raise InvalidModelException(f"""\
Transitions reference attributes which were not declared as symbols.
Missing attributes: {", ".join(map(str, missing_attrs))}""")

    # Filter to just the "in-use" compartments
    used_comps = [c for c in symbols.compartments if c.symbol in trx_comps]
    # Filter to just the "in-use" attributes
    used_attrs = [a for a in symbols.attributes if a.symbol in trx_attrs]

    def compartment_index(s: Symbol) -> int:
        return index_where(used_comps, lambda c: c.symbol == s)

    # Calc apply matrix
    num_events = Transition.event_count(transitions)
    apply_matrix = np.zeros((num_events, len(used_comps)), dtype=int)
    for eidx, e in Transition.as_events(transitions):
        apply_matrix[eidx, compartment_index(e.compartment_from)] = -1
        apply_matrix[eidx, compartment_index(e.compartment_to)] = +1

    # Calc list of events leaving each compartment (each may have 0, 1, or more)
    events_leaving_compartment = [[eidx
                                  for eidx, e in Transition.as_events(transitions)
                                  if e.compartment_from == c.symbol]
                                  for c in used_comps]

    # Calc the source compartment for each event
    source_compartment_for_event = [compartment_index(e.compartment_from)
                                    for _, e in Transition.as_events(transitions)]

    return CompartmentModel(
        transitions,
        used_comps,
        used_attrs,
        apply_matrix,
        events_leaving_compartment,
        source_compartment_for_event)
