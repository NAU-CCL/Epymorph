import re
from abc import ABC
from datetime import date
from typing import Any, Callable, ClassVar

import numpy as np
import sympy
from attr import dataclass

from epymorph.clock import Clock, Tick
from epymorph.context import SimContext
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

    def __init__(self, symbols: CompartmentSymbols, transitions: list[TransitionDef]):
        self.symbols = symbols
        self.transitions = transitions
        self.states = symbols.compartment_symbols
        self.params = symbols.attribute_symbols
        self.all_symbols = self.states + self.params
        self.attribute_getters = [a.to_getter() for a in symbols.attributes]
        self.events = [Event.for_transition(t, self.all_symbols)
                       for t in transitions]

        # TODO: check that all symbols used in the transitions are in the `symbols` object
        # state_check = set(x
        #                   for t in transitions
        #                   for x in [t.state_from, t.state_to])
        # param_check = set(cast(sympy.Symbol, x)
        #                   for t in transitions
        #                   for x in list(t.rate_expression.free_symbols)
        #                   if x not in self.states)

    def to_values(self, ctx: SimContext, tick: Tick, loc: Location) -> list[float]:
        cs = np.sum([p.compartments for p in loc.pops], axis=0)
        attribs = (f(ctx, tick, loc) for f in self.attribute_getters)
        return [*cs, *attribs]

    def rates(self, values: list[float]) -> list[float]:
        """Calculate the expected daily occurrences for each event given `values`."""
        return [e.eval(values) for e in self.events]


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


if __name__ == "__main__":
    symbols = CompartmentSymbols(
        compartments=['S', 'I', 'R'],
        attributes=[
            Param('D', 'infection_duration', shape='S'),
            Param('L', 'immunity_duration', shape='S'),
            Geo('H', 'humidity', shape='TxN')
        ])

    [[S, I, R], [D, L, H]] = symbols.all_symbols

    model = CompartmentModel(
        symbols=symbols,
        transitions=[
            Transition(S, I, rate=0.434 * S * I / (S + I + R)),
            Transition(I, R, rate=I / D),
            Transition(R, S, rate=R / L)
        ])

    # print(model.states)
    # print(model.params)
    print(model.all_symbols)
    print(list(e.name for e in model.events))
    print(model.rates([1_000_000, 1_000, 0, 4.0, 90.0, 0.532]))

    num_nodes = 3
    ctx = SimContext(
        nodes=num_nodes,
        labels=[f'node{n}' for n in range(num_nodes)],
        geo={
            'humidity': np.array([[0.5, 0.4, 0.3]], dtype=np.double)
        },
        compartments=3,
        events=3,
        param={
            'infection_duration': 4.0,
            'immunity_duration': 90.0
        },
        clock=Clock.init(date(2023, 1, 1), 50, [np.double(1)]),
        rng=np.random.default_rng(1)
    )
    loc0 = Location.initialize(0, np.array([1000, 100, 0], np.int_))
    loc1 = Location.initialize(1, np.array([2000, 0, 0], np.int_))
    loc2 = Location.initialize(2, np.array([3000, 0, 0], np.int_))
    print(model.to_values(ctx, ctx.clock.ticks[0], loc0))
    print(model.to_values(ctx, ctx.clock.ticks[0], loc1))
    print(model.to_values(ctx, ctx.clock.ticks[0], loc2))
