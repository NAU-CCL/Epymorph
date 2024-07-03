"""Defines a compartmental IPM mirroring the Pei paper's beta treatment."""
from sympy import Max, exp, log

from epymorph.compartment_model import (CompartmentModel, compartment,
                                        create_model, create_symbols, edge)
from epymorph.data import registry
from epymorph.data_shape import Shapes
from epymorph.simulation import AttributeDef


@registry.ipm('pei')
def load() -> CompartmentModel:
    """Load the 'pei' IPM."""
    symbols = create_symbols(
        compartments=[
            compartment('S'),
            compartment('I'),
            compartment('R'),
        ],
        attributes=[
            AttributeDef('infection_duration', float, Shapes.TxN),
            AttributeDef('immunity_duration', float, Shapes.TxN),
            AttributeDef('humidity', float, Shapes.TxN),
        ])

    [S, I, R] = symbols.compartment_symbols
    [D, L, H] = symbols.attribute_symbols

    beta = (exp(-180 * H + log(2.0 - 1.3)) + 1.3) / D

    # formulate N so as to avoid dividing by zero;
    # this is safe in this instance because if the denominator is zero,
    # the numerator must also be zero
    N = Max(1, S + I + R)

    return create_model(
        symbols=symbols,
        transitions=[
            edge(S, I, rate=beta * S * I / N),
            edge(I, R, rate=I / D),
            edge(R, S, rate=R / L)
        ])
