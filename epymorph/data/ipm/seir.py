# type: ignore
from sympy import Max

from epymorph.data_shape import Shapes
from epymorph.ipm.attribute import param
from epymorph.ipm.compartment_ipm import CompartmentModelIpmBuilder
from epymorph.ipm.compartment_model import (compartment, create_model,
                                            create_symbols, edge)
from epymorph.ipm.ipm import IpmBuilder


def load() -> IpmBuilder:
    symbols = create_symbols(
        compartments=[
            compartment('S'),
            compartment('E'),
            compartment('I'),
            compartment('R'),
        ],
        attributes=[
            param('beta', shape=Shapes.TxN),  # infectivity
            param('sigma', shape=Shapes.TxN),  # progression from exposed to infected
            param('gamma', shape=Shapes.TxN)  # progression from infected to recovered
        ])

    [S,E,I, R] = symbols.compartment_symbols
    [β, ξ, γ] = symbols.attribute_symbols

    N = Max(1, S + E + I + R)

    seir = create_model(
        symbols=symbols,
        transitions=[
            edge(S, E, rate=β * S * I / N),
            edge(E, I, rate=ξ * E),
            edge(I, R, rate=γ * I)
        ])

    return CompartmentModelIpmBuilder(seir)
