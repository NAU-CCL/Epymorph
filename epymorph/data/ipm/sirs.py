from sympy import Max

from epymorph.ipm.attribute import param
from epymorph.ipm.compartment_ipm import CompartmentModelIpmBuilder
from epymorph.ipm.compartment_model import (compartment, create_model,
                                            create_symbols, edge)
from epymorph.ipm.ipm import IpmBuilder


def load() -> IpmBuilder:
    symbols = create_symbols(
        compartments=[
            compartment('S'),
            compartment('I'),
            compartment('R'),
        ],
        attributes=[
            param('beta'),  # infectivity
            param('gamma'),  # progression from infected to recovered
            param('xi')  # progression from recovered to susceptible
        ])

    [S, I, R] = symbols.compartment_symbols
    [β, γ, ξ] = symbols.attribute_symbols

    # formulate N so as to avoid dividing by zero;
    # this is safe in this instance because if the denominator is zero,
    # the numerator must also be zero
    N = Max(1, S + I + R)

    sirh = create_model(
        symbols=symbols,
        transitions=[
            edge(S, I, rate=β * S * I / N),
            edge(I, R, rate=γ * I),
            edge(R, S, rate=ξ * R)
        ])

    return CompartmentModelIpmBuilder(sirh)
