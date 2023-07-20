from epymorph.ipm.attribute import param
from epymorph.ipm.compartment_ipm import CompartmentModelIpmBuilder
from epymorph.ipm.compartment_model import (compartment, create_model,
                                            create_symbols, edge, fork)
from epymorph.ipm.ipm import IpmBuilder


def load() -> IpmBuilder:
    symbols = create_symbols(
        compartments=[
            compartment('S'),
            compartment('I'),
            compartment('R'),
            compartment('H', tags=['immobile'])
        ],
        attributes=[
            param('beta'),
            param('gamma'),
            param('hospitalization_rate'),
            param('hospitalization_duration')
        ])

    [S, I, R, H] = symbols.compartment_symbols
    [β, γ, h_rate, h_dur] = symbols.attribute_symbols

    sirh = create_model(
        symbols=symbols,
        transitions=[
            edge(S, I, rate=β * S * I / (S + I + R + H)),
            fork(
                edge(I, H, rate=γ * I * h_rate),
                edge(I, R, rate=γ * I * (1 - h_rate))
            ),
            edge(H, R, rate=H / h_dur)
        ])

    return CompartmentModelIpmBuilder(sirh)
