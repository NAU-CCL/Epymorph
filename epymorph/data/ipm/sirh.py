"""Defines a compartmental IPM for a generic SIRH model."""
from sympy import Max

from epymorph.compartment_model import (CompartmentModel, compartment,
                                        create_model, create_symbols, edge,
                                        fork, param)
from epymorph.data import registry
from epymorph.data_shape import Shapes


@registry.ipm('sirh')
def load() -> CompartmentModel:
    """Load the 'sirh' IPM."""
    symbols = create_symbols(
        compartments=[
            compartment('S'),
            compartment('I'),
            compartment('R'),
            compartment('H', tags=['immobile'])
        ],
        attributes=[
            param('beta', shape=Shapes.TxN),
            param('gamma', shape=Shapes.TxN),
            param('xi', shape=Shapes.TxN),
            param('hospitalization_prob', shape=Shapes.TxN),
            param('hospitalization_duration', shape=Shapes.TxN)
        ])

    [S, I, R, H] = symbols.compartment_symbols
    [β, γ, ξ, h_prob, h_dur] = symbols.attribute_symbols

    # formulate N so as to avoid dividing by zero;
    # this is safe in this instance because if the denominator is zero,
    # the numerator must also be zero
    N = Max(1, S + I + R + H)

    return create_model(
        symbols=symbols,
        transitions=[
            edge(S, I, rate=β * S * I / N),
            fork(
                edge(I, H, rate=γ * I * h_prob),
                edge(I, R, rate=γ * I * (1 - h_prob)),
            ),
            edge(H, R, rate=H / h_dur),
            edge(R, S, rate=ξ * R),
        ])
