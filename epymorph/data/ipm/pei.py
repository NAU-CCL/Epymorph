# type: ignore (this is to silence spurious Sympy type warnings re: operators)
"""Defines a compartmental IPM mirroring the Pei paper's beta treatment."""
from sympy import Max, parse_expr

from epymorph.compartment_model import (CompartmentModel, compartment,
                                        create_model, create_symbols, edge,
                                        geo, param)
from epymorph.data import registry
from epymorph.data_shape import Shapes


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
            param('infection_duration', Shapes.TxN, float, 'D'),
            param('immunity_duration', Shapes.TxN, float, 'L'),
            geo('humidity', Shapes.TxN, float, 'H'),
        ])

    [S, I, R] = symbols.compartment_symbols
    [D, L, H] = symbols.attribute_symbols

    # BUG: TYPE-CHECKING THIS LINE IS SUPER SLOW
    # see: https://github.com/microsoft/pylance-release/issues/946 (similar?)
    # beta = (exp(-180 * H + log(2.0 - 1.3)) + 1.3) / D

    # as a workaround we can just parse the expression from a string :(
    beta = parse_expr("(exp(-180 * H + log(2.0 - 1.3)) + 1.3) / D")

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
