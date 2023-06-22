import re
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Generator, Iterable, Literal, cast

import numpy as np
import sympy
from numpy.typing import NDArray

from epymorph.clock import Tick
from epymorph.context import SimContext
from epymorph.epi import Ipm, IpmBuilder
from epymorph.util import Compartments, Events
from epymorph.world import Location

############################################################
# Model Transitions (Edge or Fork)
############################################################


@dataclass(frozen=True)
class Edge:
    state_from: sympy.Symbol
    state_to: sympy.Symbol
    rate: sympy.Expr


@dataclass(frozen=True)
class Fork:
    edges: list[Edge]
    rate: sympy.Expr
    prob: list[sympy.Expr]

    def __init__(self, edges: list[Edge]):
        # it is assumed the fork's edges are defined with complementary rate expressions, e.g.,
        # 1. \delta * E * \rho
        # 2. \delta * E * (1 - \rho)
        object.__setattr__(self, 'edges', edges)
        # the "base rate" -- how many individuals transition on any of these edges --
        # is the sum of all the edge rates (this defines the lambda for the poisson draw)
        edge_rates = [e.rate for e in self.edges]
        object.__setattr__(
            self, 'rate', sympy.simplify(sympy.Add(*edge_rates)))
        # the probability of following a particular edge is then the edge's rate divided by the base rate
        # (this defines the probability split in the eventual multinomial draw)
        object.__setattr__(self, 'prob', [sympy.simplify(r / self.rate)
                                          for r in edge_rates])


Transition = Edge | Fork


@dataclass(frozen=True)
class ModelEdge(Edge):
    rate_lambda: Any


@dataclass(frozen=True)
class ModelFork(Fork):
    size: int
    rate_lambda: Any
    prob_lambda: Any


ModelTransition = ModelEdge | ModelFork


def to_model_transition(transitions: list[Transition], args: list[sympy.Symbol]) -> list[ModelTransition]:
    def compile(t: Transition) -> ModelTransition:
        match t:
            case Edge(state_from, state_to, rate):
                rate_lambda = sympy.lambdify([args], rate)
                return ModelEdge(state_from, state_to, rate, rate_lambda)
            case Fork(edges, rate, prob):
                size = len(edges)
                rate_lambda = sympy.lambdify([args], rate)
                prob_lambda = sympy.lambdify([args], prob)
                return ModelFork(edges, rate, prob, size, rate_lambda, prob_lambda)
    return list(map(compile, transitions))


def eval_rates(ctx: SimContext, transitions: list[ModelTransition], params: list[Any]) -> NDArray[np.int_]:
    occurrences = np.zeros(ctx.events, dtype=int)
    index = 0
    for t in transitions:
        match t:
            case ModelEdge(_, _, _, rate_lambda):
                rate = rate_lambda(params)
                occurrences[index] = ctx.rng.poisson(rate)
                index += 1
            case ModelFork(_, _, _, size, rate_lambda, prob_lambda):
                rate = rate_lambda(params)
                base = ctx.rng.poisson(rate)
                prob = prob_lambda(params)
                stop = index + size
                occurrences[index:stop] = ctx.rng.multinomial(base, prob)
                index = stop
    return occurrences


def edge_iterator(transitions: Iterable[Transition]) -> Generator[tuple[(int, Edge)], None, None]:
    index = 0
    for t in transitions:
        match t:
            case Edge(_, _, _) as e:
                yield (index, e)
                index += 1
            case Fork(edges):
                for e in edges:
                    yield (index, e)
                    index += 1


def edge_count(transitions: Iterable[Transition]) -> int:
    count = 0
    for _ in edge_iterator(transitions):
        count += 1
    return count


def extract_symbols(transitions: Iterable[Transition]) -> set[sympy.Symbol]:
    syms = set[sympy.Symbol]()

    def add_expr_syms(expr: sympy.Expr):
        for x in expr.free_symbols:
            if isinstance(x, sympy.Symbol):
                syms.add(x)

    for t in transitions:
        match t:
            case Edge(_, _, rate):
                add_expr_syms(rate)
            case Fork(_, rate, probs):
                add_expr_syms(rate)
                for p in probs:
                    add_expr_syms(p)

    return syms


############################################################
# Attributes
############################################################
# IPMs can use parameters of any of these shapes
# (where A stands for an "arbitrary" integer index, 0 or more)
# S is a single scalar value
# T is the number of ticks
# N is the number of nodes
# ---
# A
# S
# T
# N
# TxA
# NxA
# TxN
# TxNxA
shape_regex = re.compile(r"^A|[STN]|[TN]xA|TxN(xA)?$"
                         .replace("A", "(0|[1-9][0-9]*)"))

# MMs can use pairwise data, and so would add the following cases:
# NxN
# NxNxA
# TxNxN
# TxNxNxA


def test_shape():
    # TODO: make this a unit test
    lines = [
        # Success cases:
        'S',
        '0',
        '2',
        '3',
        '13',
        'T',
        'Tx9',
        'N',
        'Nx20',
        'TxN',
        'TxNx2',
        'Tx0',
        'Tx1',
        'Nx1',
        'TxNx1',
        # Failure cases:
        'NxN',
        'NxNx32',
        'TxNxN',
        'TxNxNx4',
        'A',
        '3BC',
        'Tx3N',
        '3T',
        'T3',
        'N3T',
        'TxT',
        'NxN3',
        '3TxN',
        'TxN3T',
        'Tx3T',
        'NTxN',
        'NxTxN',
    ]

    for line in lines:
        if shape_regex.fullmatch(line):
            print(f"Matched: {line}")
        else:
            print(f"Not matched: {line}")


AttributeGetter = Callable[[SimContext, Tick, Location], float]


@dataclass(frozen=True)
class Attribute(ABC):
    symbol_name: str
    attribute_name: str
    shape: str
    dtype: Literal['int'] | Literal['float']

    data_source: ClassVar[str]
    """Which SimContext dict is this attribute drawing from?"""

    def __post_init__(self):
        if not shape_regex.fullmatch(self.shape):
            raise Exception(f"Not a valid parameter shape: {self.shape}")

    def to_getter(self) -> AttributeGetter:
        x = self.attribute_name
        match self.shape:
            case "S":
                return lambda ctx, tick, loc: getattr(ctx, self.data_source)[x]
            case "T":
                return lambda ctx, tick, loc: getattr(ctx, self.data_source)[x][tick.day]
            case "N":
                return lambda ctx, tick, loc: getattr(ctx, self.data_source)[x][loc.index]
            case "TxN":
                return lambda ctx, tick, loc: getattr(ctx, self.data_source)[x][tick.day, loc.index]
            case shape:
                # shape is one of: A, TxA, NxA, TxNxA (where A is an index)
                # unfortunately Python's structural pattern matching doesn't allow regex yet
                index_regex = re.compile(r"^(.*?)([0-9]+)$")
                match = index_regex.match(shape)
                if match is None:
                    raise Exception(f"Unsupported shape: {shape}")

                prefix, arbitrary_index = match.groups()
                match prefix:
                    case "":
                        return lambda ctx, tick, loc: getattr(ctx, self.data_source)[x][arbitrary_index]
                    case "Tx":
                        return lambda ctx, tick, loc: getattr(ctx, self.data_source)[x][tick.day, arbitrary_index]
                    case "Nx":
                        return lambda ctx, tick, loc: getattr(ctx, self.data_source)[x][loc.index, arbitrary_index]
                    case "TxNx":
                        return lambda ctx, tick, loc: getattr(ctx, self.data_source)[x][tick.day, loc.index, arbitrary_index]
                    case _:
                        raise Exception(f"Unsupported shape: {shape}")


@dataclass(frozen=True)
class Geo(Attribute):
    data_source = 'geo'


@dataclass(frozen=True)
class Param(Attribute):
    data_source = 'param'

############################################################
# Compartment Symbols and Model
############################################################


def _to_symbols(names: list[str]) -> list[sympy.Symbol]:
    return sympy.symbols(names)  # type: ignore


def _to_symbol(name: str) -> sympy.Symbol:
    return sympy.symbols(name)  # type: ignore


@dataclass(frozen=True)
class CompartmentSymbols:
    compartments: list[sympy.Symbol]
    attributes: dict[sympy.Symbol, Attribute]
    all_symbols: list[sympy.Symbol]

    @classmethod
    def create(cls, compartments: list[str], attributes: list[Attribute]):
        compartment_symbols = _to_symbols(compartments)
        attribs_dict = dict([(_to_symbol(a.symbol_name), a)
                             for a in attributes])
        all_symbols = compartment_symbols + list(attribs_dict.keys())
        return cls(compartment_symbols, attribs_dict, all_symbols)


class InvalidModelException(Exception):
    pass


class CompartmentModel:
    transitions: list[Transition]
    transitions_args: list[sympy.Symbol]
    num_states: int
    num_events: int
    states: list[sympy.Symbol]
    attributes: list[Attribute]
    attribute_getters: list[AttributeGetter]

    # a matrix defining how each event impacts each compartment (subtracting or adding individuals)
    apply_matrix: NDArray[np.int_]
    # mapping from compartment index to the list of event indices which source from that compartment
    source_events: list[list[int]]
    # mapping from event index to the compartment index it sources from
    event_source: list[int]

    def __init__(self, symbols: CompartmentSymbols, transitions: list[Transition]):
        self.transitions = transitions
        self.num_states = len(symbols.compartments)
        self.num_events = edge_count(transitions)
        self.states = symbols.compartments

        # Only keep attributes which are used in the transition expressions.
        # It's possible the user defined more than necessary.
        attribs_used = extract_symbols(transitions)
        self.attributes = [a for s, a in symbols.attributes.items()
                           if s in attribs_used]
        attribute_symbols = [s for s in symbols.attributes.keys()
                             if s in attribs_used]
        self.attribute_getters = [a.to_getter() for a in self.attributes]
        self.transitions_args = self.states + attribute_symbols

        # Compute useful metadata about the transitions.
        self.apply_matrix = create_apply_matrix(self)
        src = [list[int]() for _ in range(self.num_states)]
        for eidx, edge in edge_iterator(transitions):
            sidx = self.states.index(edge.state_from)
            src[sidx].append(eidx)
        self.source_events = src
        self.event_source = [self.states.index(edge.state_from)
                             for _, edge in edge_iterator(transitions)]


def validate_model(model: CompartmentModel) -> None:
    # Check: all states in transitions must be declared in CompartmentSymbols
    trx_states = set(x
                     for _, t in edge_iterator(model.transitions)
                     for x in [t.state_from, t.state_to])
    missing_states = trx_states.difference(model.states)
    if len(missing_states) > 0:
        raise InvalidModelException(f"""\
Compartment model specification is incorrect.
Transitions reference compartments which were not declared in CompartmentSymbols.
Missing states: {", ".join(map(str, missing_states))}""")

    # Check: all attributes in transitions must be declared in CompartmentSymbols.
    trx_attribs = extract_symbols(model.transitions).difference(model.states)
    missing_attribs = trx_attribs.difference(model.transitions_args)
    if len(missing_attribs) > 0:
        raise InvalidModelException(f"""\
Compartment model specification is incorrect.
Transitions reference attributes which were not declared in CompartmentSymbols.
Missing attributes: {", ".join(map(str, missing_attribs))}""")


def create_apply_matrix(model: CompartmentModel) -> NDArray[np.int_]:
    transitions = model.transitions
    states = model.states
    mat = np.zeros((model.num_events, model.num_states), dtype=np.int_)
    for event_idx, t in edge_iterator(transitions):
        mat[event_idx, states.index(t.state_from)] = -1
        mat[event_idx, states.index(t.state_to)] = +1
    return mat

############################################################
# IPM Implementation
############################################################


class CompartmentalIpm(Ipm):
    ctx: SimContext
    model: CompartmentModel
    model_transitions: list[ModelTransition]

    def __init__(self, ctx: SimContext, model: CompartmentModel):
        self.ctx = ctx
        self.model = model
        self.model_transitions = to_model_transition(
            model.transitions, model.transitions_args)

    def _event_args(self, loc: Location, tick: Tick) -> list[Any]:
        # Assemble rate function arguments for this location/tick.
        all_pops = np.array([p.compartments for p in loc.pops], dtype=int)
        effective = np.sum(all_pops, axis=0)
        attribs = (f(self.ctx, tick, loc)
                   for f in self.model.attribute_getters)
        return [*effective, *attribs]

    def events(self, loc: Location, tick: Tick) -> Events:
        rate_args = self._event_args(loc, tick)
        # Calculate how many events we expect to happen this tick.
        erates = eval_rates(self.ctx, self.model_transitions, rate_args)
        expect = self.ctx.rng.poisson(erates * tick.tau)
        actual = expect.copy()

        # Check for event overruns leaving each compartment and reduce counts.
        available = loc.compartment_totals
        for sidx, eidxs in enumerate(self.model.source_events):
            n = len(eidxs)
            if n == 0:
                # Compartment has no outward edges; nothing to do here.
                continue
            elif n == 1:
                # Compartment only has one outward edge; just "min".
                eidx = eidxs[0]
                actual[eidx] = min(expect[eidx], available[sidx])
            elif n == 2:
                # Compartment has two outward edges:
                # use hypergeo to select which events "actually" happened.
                eidx0 = eidxs[0]
                eidx1 = eidxs[1]
                desired0 = expect[eidx0]
                desired1 = expect[eidx1]
                av = available[sidx]
                if desired0 + desired1 > av:
                    drawn0 = self.ctx.rng.hypergeometric(
                        desired0, desired1, av)
                    actual[eidx0] = drawn0
                    actual[eidx1] = av - drawn0
            else:
                # Compartment has more than two outwards edges:
                # use multivariate hypergeometric to select which events "actually" happened.
                desired = expect[eidxs]
                av = available[sidx]
                if np.sum(desired) > av:
                    actual[eidxs] = self.ctx.rng.multivariate_hypergeometric(
                        desired, av)
        return actual

    def apply_events(self, loc: Location, es: NDArray[np.int_]) -> None:
        compartments = np.array([pop.compartments for pop in loc.pops])
        occurrences_by_pop = np.zeros(
            (self.ctx.nodes, self.ctx.events), dtype=int)
        # For each event, redistribute across loc's pops
        for eidx, occur in enumerate(es):
            sidx = self.model.event_source[eidx]
            occurrences_by_pop[:, eidx] = self.ctx.rng.multivariate_hypergeometric(
                compartments[:, sidx], occur)
        # Now that events are assigned to pops, update pop compartments using apply matrix.
        for pidx, pop in enumerate(loc.pops):
            deltas = np.matmul(
                occurrences_by_pop[pidx], self.model.apply_matrix)
            pop.compartments += deltas


class CompartmentalIpmBuilder(IpmBuilder):
    model: CompartmentModel

    def __init__(self, model: CompartmentModel):
        self.model = model
        super().__init__(model.num_states, model.num_events)

    def build(self, ctx: SimContext) -> Ipm:
        return CompartmentalIpm(ctx, self.model)

    def verify(self, ctx: SimContext) -> None:
        errors = list[str]()
        for a in self.model.attributes:
            source: dict[str, Any] = getattr(ctx, a.data_source)
            # does attribute exist?
            if not a.attribute_name in source:
                errors.append(
                    f"Attribute {a.attribute_name} missing from {a.data_source}.")
                # if missing, no need to do the rest of the checks
                continue

            data = source[a.attribute_name]

            if a.shape == "S":
                # check scalar values
                # int values must be specified as an int
                # but float values may be specified as a float or an int
                exp_type = int if a.dtype == 'int' else (int, float)
                if not isinstance(data, exp_type):
                    errors.append(
                        f"Attribute {a.attribute_name} was expected to be a scalar {a.dtype}.")
            else:
                # check numpy array values
                exp_type = np.dtype(
                    np.int_) if a.dtype == 'int' else np.dtype(np.double)
                if not isinstance(data, np.ndarray):
                    errors.append(
                        f"Attribute {a.attribute_name} was expected to be an array.")
                elif data.dtype.type != exp_type.type:
                    errors.append(
                        f"Attribute {a.attribute_name} was expected to be an array of type {a.dtype}."
                    )
                # TODO: check shape

        if len(errors) > 0:
            raise Exception(
                "IPM attribute requirements were not met. See errors:" + "".join(f"\n- {e}" for e in errors))

    def initialize_compartments(self, ctx: SimContext) -> list[Compartments]:
        # Initial compartments based on population (C0)
        population = ctx.geo['population']
        # With a seeded infection (C1) in one location
        si = ctx.param['infection_seed_loc']
        sn = ctx.param['infection_seed_size']
        cs = np.zeros((ctx.nodes, ctx.compartments), dtype=np.int_)
        cs[:, 0] = population
        cs[si, 0] -= sn
        cs[si, 1] += sn
        return list(cs)
