"""Defines a copmartmental IPM mirroring the SPARSEMOD COVID model."""

from sympy import Max

from epymorph.attribute import AttributeDef
from epymorph.compartment_model import CompartmentModel, compartment, edge, fork
from epymorph.data_shape import Shapes


class SparseMod(CompartmentModel):
    """A model similar to one used in sparsemod."""

    compartments = [
        compartment("S", description="susceptible"),
        compartment("E", description="exposed"),
        compartment("I_a", description="infected asymptomatic"),
        compartment("I_p", description="infected presymptomatic"),
        compartment("I_s", description="infected symptomatic"),
        compartment("I_b", description="infected bed-rest"),
        compartment("I_h", description="infected hospitalized"),
        compartment("I_c1", description="infected in ICU"),
        compartment("I_c2", description="infected in ICU Step-Down"),
        compartment("D", description="deceased"),
        compartment("R", description="recovered"),
    ]

    requirements = [
        AttributeDef("beta", type=float, shape=Shapes.TxN),
        AttributeDef("omega_1", type=float, shape=Shapes.TxN),
        AttributeDef("omega_2", type=float, shape=Shapes.TxN),
        AttributeDef("delta_1", type=float, shape=Shapes.TxN),
        AttributeDef("delta_2", type=float, shape=Shapes.TxN),
        AttributeDef("delta_3", type=float, shape=Shapes.TxN),
        AttributeDef("delta_4", type=float, shape=Shapes.TxN),
        AttributeDef("delta_5", type=float, shape=Shapes.TxN),
        AttributeDef("gamma_a", type=float, shape=Shapes.TxN),
        AttributeDef("gamma_b", type=float, shape=Shapes.TxN),
        AttributeDef("gamma_c", type=float, shape=Shapes.TxN),
        AttributeDef("rho_1", type=float, shape=Shapes.TxN),
        AttributeDef("rho_2", type=float, shape=Shapes.TxN),
        AttributeDef("rho_3", type=float, shape=Shapes.TxN),
        AttributeDef("rho_4", type=float, shape=Shapes.TxN),
        AttributeDef("rho_5", type=float, shape=Shapes.TxN),
    ]

    def edges(self, symbols):
        [S, E, I_a, I_p, I_s, I_b, I_h, I_c1, I_c2, D, R] = symbols.all_compartments
        [
            beta,
            omega_1,
            omega_2,
            delta_1,
            delta_2,
            delta_3,
            delta_4,
            delta_5,
            gamma_a,
            gamma_b,
            gamma_c,
            rho_1,
            rho_2,
            rho_3,
            rho_4,
            rho_5,
        ] = symbols.all_requirements

        # formulate the divisor so as to avoid dividing by zero;
        # this is safe in this instance becase if the denominator is zero,
        # the numerator must also be zero
        N = Max(1, S + E + I_a + I_p + I_s + I_b + I_h + I_c1 + I_c2 + R)
        lambda_1 = (omega_1 * I_a + I_p + I_s + I_b + omega_2 * (I_h + I_c1 + I_c2)) / N

        return [
            edge(S, E, rate=beta * lambda_1 * S),
            fork(
                edge(E, I_a, rate=E * delta_1 * rho_1),
                edge(E, I_p, rate=E * delta_1 * (1 - rho_1)),
            ),
            edge(I_p, I_s, rate=I_p * delta_2),
            fork(
                edge(I_s, I_h, rate=I_s * delta_3 * rho_2),
                edge(I_s, I_c1, rate=I_s * delta_3 * rho_3),
                edge(I_s, I_b, rate=I_s * delta_3 * (1 - rho_2 - rho_3)),
            ),
            fork(
                edge(I_h, I_c1, rate=I_h * delta_4 * rho_4),
                edge(I_h, R, rate=I_h * delta_4 * (1 - rho_4)),
            ),
            fork(
                edge(I_c1, D, rate=I_c1 * delta_5 * rho_5),
                edge(I_c1, I_c2, rate=I_c1 * delta_5 * (1 - rho_5)),
            ),
            edge(I_a, R, rate=I_a * gamma_a),
            edge(I_b, R, rate=I_b * gamma_b),
            edge(I_c2, R, rate=I_c2 * gamma_c),
        ]
