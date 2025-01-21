"""Defines a compartmental IPM mirroring the Pei paper's beta treatment."""

from sympy import Max, exp, log

from epymorph.attribute import AttributeDef
from epymorph.compartment_model import CompartmentModel, compartment, edge
from epymorph.data_shape import Shapes


class Pei(CompartmentModel):
    """The 'pei' IPM: an SIRS model driven by humidity."""

    compartments = [
        compartment("S"),
        compartment("I"),
        compartment("R"),
    ]

    requirements = [
        AttributeDef("infection_duration", float, Shapes.TxN),
        AttributeDef("immunity_duration", float, Shapes.TxN),
        AttributeDef("humidity", float, Shapes.TxN),
    ]

    def edges(self, symbols):
        [S, I, R] = symbols.all_compartments
        [D, L, H] = symbols.all_requirements

        beta = (exp(-180 * H + log(2.0 - 1.3)) + 1.3) / D

        # formulate N so as to avoid dividing by zero;
        # this is safe in this instance because if the denominator is zero,
        # the numerator must also be zero
        N = Max(1, S + I + R)

        return [
            edge(S, I, rate=beta * S * I / N),
            edge(I, R, rate=I / D),
            edge(R, S, rate=R / L),
        ]
