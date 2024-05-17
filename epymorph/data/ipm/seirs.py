from sympy import Max

from epymorph.compartment_model import (CompartmentModel, compartment,
                                        create_model, create_symbols, edge)
from epymorph.data import registry
from epymorph.data_shape import Shapes
from epymorph.simulation import params_attrib as param


@registry.ipm('seirs')
def load() -> CompartmentModel:
    symbols = create_symbols(
        compartments=[
            compartment('S'),
            compartment('E'),
            compartment('I'),
            compartment('R'),
        ],
        attributes=[
            param('beta', dtype=float, shape=Shapes.TxN,
                  comment='infectivity'),
            param('sigma', dtype=float, shape=Shapes.TxN,
                  comment='progression from exposed to infected'),
            param('gamma', dtype=float, shape=Shapes.TxN,
                  comment='progression from infected to recovered'),
            param('xi', dtype=float, shape=Shapes.TxN,
                  comment='progression from recovered to susceptible'),
        ])

    [S, E, I, R] = symbols.compartment_symbols
    [β, σ, γ, ξ] = symbols.attribute_symbols

    N = Max(1, S + E + I + R)

    return create_model(
        symbols=symbols,
        transitions=[
            edge(S, E, rate=β * S * I / N),
            edge(E, I, rate=σ * E),
            edge(I, R, rate=γ * I),
            edge(R, S, rate=ξ * R)
        ])
