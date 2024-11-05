from sympy import Max

from epymorph.compartment_model import CompartmentModel, compartment, edge
from epymorph.data import registry
from epymorph.data_shape import Shapes
from epymorph.simulation import AttributeDef


@registry.ipm("seir")
class Seir(CompartmentModel):
    """A basic SEIR model."""

    compartments = [
        compartment("S"),
        compartment("E"),
        compartment("I"),
        compartment("R"),
        # compartment('D')
    ]

    requirements = [
        AttributeDef("beta", type=float, shape=Shapes.TxN, comment="infectivity"),
        AttributeDef(
            "eta",
            type=float,
            shape=Shapes.TxN,
            comment="progression from exposed to infected",
        ),
        AttributeDef("gamma", type=float, shape=Shapes.TxN, comment="recovery rate"),
        AttributeDef("q", type=float, shape=Shapes.TxN, comment="recovery rate"),
    ]

    def edges(self, symbols):
        [S, E, I, R] = symbols.all_compartments
        [β, ξ, γ, q] = symbols.all_requirements

        # formulate N so as to avoid dividing by zero;
        # this is safe in this instance because if the denominator is zero,
        # the numerator must also be zero
        N = Max(1, S + E + I + R)

        return [
            edge(S, E, rate=β * ((E + q * I) / N) * S),
            edge(E, I, rate=ξ * E),
            edge(E, R, rate=γ * E),
            edge(I, R, rate=γ * I),
        ]
