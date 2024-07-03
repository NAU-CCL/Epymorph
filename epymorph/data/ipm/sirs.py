"""Defines a compartmental IPM for a generic SIRS model."""
from sympy import Max

from epymorph.compartment_model import (CompartmentModel, compartment,
                                        create_model, create_symbols, edge)
from epymorph.data import registry
from epymorph.data_shape import Shapes
from epymorph.simulation import AttributeDef


@registry.ipm('sirs')
def load() -> CompartmentModel:
    """Load the 'sirs' IPM."""
    symbols = create_symbols(
        compartments=[
            compartment('S'),
            compartment('I'),
            compartment('R'),
        ],
        attributes=[
            AttributeDef('beta', type=float, shape=Shapes.TxN,
                         comment='infectivity'),
            AttributeDef('gamma', type=float, shape=Shapes.TxN,
                         comment='progression from infected to recovered'),
            AttributeDef('xi', type=float, shape=Shapes.TxN,
                         comment='progression from recovered to susceptible'),
        ])

    [S, I, R] = symbols.compartment_symbols
    [β, γ, ξ] = symbols.attribute_symbols

    # formulate N so as to avoid dividing by zero;
    # this is safe in this instance because if the denominator is zero,
    # the numerator must also be zero
    N = Max(1, S + I + R)

    return create_model(
        symbols=symbols,
        transitions=[
            edge(S, I, rate=β * S * I / N),
            edge(I, R, rate=γ * I),
            edge(R, S, rate=ξ * R)
        ])
