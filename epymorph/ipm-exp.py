import re
from abc import ABC
from datetime import date
from typing import Any, Callable, ClassVar, Generator, cast

import numpy as np
import sympy
from attr import dataclass
from numpy.typing import NDArray

from epymorph.clock import Clock, Tick
from epymorph.context import SimContext
from epymorph.data import geo_library, mm_library
from epymorph.epi import Ipm, IpmBuilder
from epymorph.run import plot_event, plot_pop
from epymorph.simulation import Simulation
from epymorph.util import Compartments, Events
from epymorph.world import Location


@dataclass(frozen=True)
class Transition:
    state_from: sympy.Symbol
    state_to: sympy.Symbol
    rate: sympy.Expr


# @dataclass(frozen=True)
# class Group:
#     transitions: list[Transition]


TransitionDef = Transition

# TransitionDef = Transition | Group


def event_length(transitions: list[TransitionDef]) -> int:
    # TODO: this won't be sufficient when Groups are implemented
    return len(transitions)


def event_iterator(transitions: list[TransitionDef]) -> Generator[tuple[(int, sympy.Symbol, sympy.Symbol)], None, None]:
    index = 0
    for e in transitions:
        match e:
            case Transition(state_from, state_to, _):
                yield (index, state_from, state_to)
                index += 1
            # case Group(state_from, sub_events, _):
            #     for e in sub_events:
            #         yield (index, state_from, e.state_to)
            #         index += 1


@dataclass
class Event:
    name: str
    trx: Transition
    rate_lambda: Any

    @classmethod
    def for_transition(cls, trx: Transition, params: list[sympy.Symbol]):
        name = f"{trx.state_from}→{trx.state_to}"
        rate_lambda = sympy.lambdify([params], trx.rate)
        return cls(name, trx, rate_lambda)

    def eval(self, params: list[float]) -> float:
        return float(self.rate_lambda(params))


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

AttributeGetter = Callable[[SimContext, Tick, Location], float]


@dataclass(frozen=True)
class Attribute(ABC):
    symbol_name: str
    attribute_name: str
    shape: str

    _data_source: ClassVar[str]
    """Which SimContext dict is this attribute drawing from?"""

    def __post_init__(self):
        if not shape_regex.fullmatch(self.shape):
            raise Exception(f"Not a valid parameter shape: {self.shape}")

    def to_getter(self) -> AttributeGetter:
        x = self.attribute_name
        match self.shape:
            case "S":
                return lambda ctx, tick, loc: getattr(ctx, self._data_source)[x]
            case "T":
                return lambda ctx, tick, loc: getattr(ctx, self._data_source)[x][tick.day]
            case "N":
                return lambda ctx, tick, loc: getattr(ctx, self._data_source)[x][loc.index]
            case "TxN":
                return lambda ctx, tick, loc: getattr(ctx, self._data_source)[x][tick.day, loc.index]
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
                        return lambda ctx, tick, loc: getattr(ctx, self._data_source)[x][arbitrary_index]
                    case "Tx":
                        return lambda ctx, tick, loc: getattr(ctx, self._data_source)[x][tick.day, arbitrary_index]
                    case "Nx":
                        return lambda ctx, tick, loc: getattr(ctx, self._data_source)[x][loc.index, arbitrary_index]
                    case "TxNx":
                        return lambda ctx, tick, loc: getattr(ctx, self._data_source)[x][tick.day, loc.index, arbitrary_index]
                    case _:
                        raise Exception(f"Unsupported shape: {shape}")


@dataclass(frozen=True)
class Geo(Attribute):
    _data_source = 'geo'


@dataclass(frozen=True)
class Param(Attribute):
    _data_source = 'param'


@dataclass(frozen=True)
class CompartmentSymbols:
    compartments: list[str]
    attributes: list[Attribute]

    @property
    def all_symbols(self) -> list[list[sympy.Symbol]]:
        return [self.compartment_symbols, self.attribute_symbols]

    @property
    def compartment_symbols(self) -> list[sympy.Symbol]:
        return sympy.symbols(self.compartments)

    @property
    def attribute_symbols(self) -> list[sympy.Symbol]:
        symbol_names = [a.symbol_name for a in self.attributes]
        return sympy.symbols(symbol_names)


class CompartmentModel:
    symbols: CompartmentSymbols
    transitions: list[TransitionDef]
    states: list[sympy.Symbol]
    params: list[sympy.Symbol]
    all_symbols: list[sympy.Symbol]
    attribute_getters: list[AttributeGetter]
    events: list[Event]
    num_events: int

    def __init__(self, symbols: CompartmentSymbols, transitions: list[TransitionDef]):
        self.symbols = symbols
        self.transitions = transitions
        self.states = symbols.compartment_symbols
        self.params = symbols.attribute_symbols
        self.all_symbols = self.states + self.params
        self.attribute_getters = [a.to_getter() for a in symbols.attributes]
        self.events = [Event.for_transition(t, self.all_symbols)
                       for t in transitions]
        self.num_events = event_length(transitions)

        # TODO: check that all symbols used in the transitions are in the `symbols` object
        # state_check = set(x
        #                   for t in transitions
        #                   for x in [t.state_from, t.state_to])
        # param_check = set(cast(sympy.Symbol, x)
        #                   for t in transitions
        #                   for x in list(t.rate_expression.free_symbols)
        #                   if x not in self.states)

    def to_values(self, ctx: SimContext, tick: Tick, loc: Location) -> list[float]:
        all_pops = np.array([p.compartments for p in loc.pops], dtype=int)
        effective = np.sum(all_pops, axis=0)
        attribs = (f(ctx, tick, loc) for f in self.attribute_getters)
        return [*effective, *attribs]

    def rates(self, values: list[float]) -> NDArray[np.double]:
        """Calculate the expected daily occurrences for each event given `values`."""
        return np.fromiter((e.eval(values) for e in self.events), dtype=float, count=self.num_events)

    def to_apply_matrix(self) -> NDArray[np.int_]:
        num_events = event_length(self.transitions)
        num_states = len(self.states)
        mat = np.zeros((num_events, num_states), dtype=np.int_)
        for event_idx, state_from, state_to in event_iterator(self.transitions):
            mat[event_idx, self.states.index(state_from)] = -1
            mat[event_idx, self.states.index(state_to)] = +1
        return mat


class CompartmentalIpm(Ipm):
    ctx: SimContext
    model: CompartmentModel

    # a matrix defining how each event impacts each compartment (subtracting or adding individuals)
    _apply_matrix: NDArray[np.int_]
    # mapping from compartment index to the list of event indices which source from that compartment
    _source_events: list[list[int]]
    # mapping from event index to the compartment index it sources from
    _event_source: list[int]

    def __init__(self, ctx: SimContext, model: CompartmentModel):
        self.ctx = ctx
        self.model = model

        self._apply_matrix = model.to_apply_matrix()
        # compute which events come from which source, and vice versa
        src = [list[int]() for _ in range(len(model.states))]
        evt = [0 for _ in range(len(model.events))]
        for eidx, state_from, _ in event_iterator(model.transitions):
            sidx = model.states.index(state_from)
            src[sidx].append(eidx)
            evt[eidx] = sidx
        self._source_events = src
        self._event_source = evt

    def events(self, loc: Location, tick: Tick) -> Events:
        # First calculate how many events we expect to happen this tick.
        values = self.model.to_values(self.ctx, tick, loc)
        erates = self.model.rates(values)
        expect = self.ctx.rng.poisson(erates * tick.tau)

        # TODO: Check for event overruns leaving each compartment and reduce counts.
        actual = expect
        return actual

    def apply_events(self, loc: Location, es: Events) -> None:
        cs = np.array([pop.compartments for pop in loc.pops])
        es_by_pop = np.zeros((self.ctx.nodes, self.ctx.events), dtype=int)
        # For each event, redistribute across loc's pops
        for eidx, occur in enumerate(es):
            sidx = self._event_source[eidx]
            es_by_pop[:, eidx] = self.ctx.rng.multivariate_hypergeometric(
                cs[:, sidx], occur)
        # Now that events are assigned to pops, update pop compartments using apply matrix.
        for pidx, pop in enumerate(loc.pops):
            deltas = np.matmul(es_by_pop[pidx], self._apply_matrix)
            pop.compartments += deltas


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


class Builder(IpmBuilder):
    def __init__(self):
        super().__init__(num_compartments=3, num_events=3)

    def build(self, ctx: SimContext) -> Ipm:
        cs = CompartmentSymbols(
            compartments=['S', 'I', 'R'],
            attributes=[
                Param('D', 'infection_duration', shape='S'),
                Param('L', 'immunity_duration', shape='S'),
                Geo('H', 'humidity', shape='TxN')
            ])

        [[S, I, R], [D, L, H]] = cs.all_symbols

        beta = (sympy.exp(-180 * H + sympy.log(2 - 1.3)) + 1.3) / D

        model = CompartmentModel(
            symbols=cs,
            transitions=[
                Transition(S, I, rate=beta * S * I / (S + I + R)),
                Transition(I, R, rate=I / D),
                Transition(R, S, rate=R / L)
            ])

        return CompartmentalIpm(ctx, model)

    def verify(self, ctx: SimContext) -> None:
        pass

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


if __name__ == "__main__":
    builder = Builder()

    # num_nodes = 3
    # ctx = SimContext(
    #     nodes=num_nodes,
    #     labels=[f'node{n}' for n in range(num_nodes)],
    #     geo={
    #         'humidity': np.array([[0.5, 0.4, 0.3]], dtype=np.double)
    #     },
    #     compartments=3,
    #     events=3,
    #     param={
    #         'infection_duration': 4.0,
    #         'immunity_duration': 90.0
    #     },
    #     clock=Clock.init(date(2023, 1, 1), 50, [np.double(1)]),
    #     rng=np.random.default_rng(1)
    # )

    # ipm = cast(CompartmentalIpm, builder.build(ctx))
    # model = ipm.model

    # print(model.states)
    # print(model.params)
    # print(model.all_symbols)
    # print(list(e.name for e in model.events))
    # print(model.rates([1_000_000, 1_000, 0, 4.0, 90.0, 0.532]))

    # loc0 = Location.initialize(0, np.array([1000, 100, 0], np.int_))
    # loc1 = Location.initialize(1, np.array([2000, 0, 0], np.int_))
    # loc2 = Location.initialize(2, np.array([3000, 0, 0], np.int_))
    # print(model.to_values(ctx, ctx.clock.ticks[0], loc0))
    # print(model.to_values(ctx, ctx.clock.ticks[0], loc1))
    # print(model.to_values(ctx, ctx.clock.ticks[0], loc2))

    sim = Simulation(
        geo=geo_library['pei'](),
        ipm_builder=builder,
        mvm_builder=mm_library['pei']()
    )

    out = sim.run(
        param={
            'theta': 0.1,
            'move_control': 0.9,
            'infection_duration': 4.0,
            'immunity_duration': 90.0,
            'infection_seed_loc': 0,
            'infection_seed_size': 10_000
        },
        start_date=date(2015, 1, 1),
        duration_days=150,
    )

    # plot_pop(out, 0)
    plot_event(out, 0)
