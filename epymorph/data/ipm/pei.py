# type: ignore
from sympy import Max, exp, log, parse_expr

from epymorph.data_shape import Shapes
from epymorph.ipm.attribute import geo, param
from epymorph.ipm.compartment_ipm import CompartmentModelIpmBuilder
from epymorph.ipm.compartment_model import (create_model, create_symbols, edge,
                                            quick_compartments)
from epymorph.ipm.ipm import IpmBuilder


def load() -> IpmBuilder:
    symbols = create_symbols(
        compartments=quick_compartments('S I R'),
        attributes=[
            param('D', 'infection_duration', Shapes.TxN),
            param('L', 'immunity_duration', Shapes.TxN),
            geo('H', 'humidity', Shapes.TxN),
        ])

    [S, I, R] = symbols.compartment_symbols
    [D, L, H] = symbols.attribute_symbols

    # TODO: TYPE-CHECKING THIS LINE IS SUPER SLOW
    # see: https://github.com/microsoft/pylance-release/issues/946 (similar?)
    # beta = (exp(-180 * H + log(2.0 - 1.3)) + 1.3) / D

    # as a workaround we can just parse the expression from a string :(
    beta = parse_expr("(exp(-180 * H + log(2.0 - 1.3)) + 1.3) / D")

    # formulate N so as to avoid dividing by zero;
    # this is safe in this instance because if the denominator is zero,
    # the numerator must also be zero
    N = Max(1, S + I + R)

    pei = create_model(
        symbols=symbols,
        transitions=[
            edge(S, I, rate=beta * S * I / N),
            edge(I, R, rate=I / D),
            edge(R, S, rate=R / L)
        ])

    return CompartmentModelIpmBuilder(pei)
