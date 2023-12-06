# type: ignore
"""Defines a copmartmental IPM mirroring the SPARSEMOD COVID model."""
from sympy import Max

from epymorph.compartment_model import (CompartmentModel, compartment,
                                        create_model, create_symbols, edge,
                                        fork, param)
from epymorph.data import registry
from epymorph.data_shape import Shapes


@registry.ipm('sparsemod')
def load() -> CompartmentModel:
    """Load the 'sparsemod' IPM."""
    symbols = create_symbols(
        compartments=[
            compartment('S', description='susceptible'),
            compartment('E', description='exposed'),
            compartment('Ia', description='infected asymptomatic'),
            compartment('Ip', description='infected presymptomatic'),
            compartment('Is', description='infected symptomatic'),
            compartment('Ib', description='infected bed-rest'),
            compartment('Ih', description='infected hospitalized'),
            compartment('Ic1', description='infected in ICU'),
            compartment('Ic2', description='infected in ICU Step-Down'),
            compartment('D', description='deceased'),
            compartment('R', description='recovered')
        ],
        attributes=[
            param('beta', Shapes.TxN, symbolic_name='beta_1'),
            param('omega', Shapes.TxNxA(0), symbolic_name='omega_1'),
            param('omega', Shapes.TxNxA(1), symbolic_name='omega_2'),
            param('delta', Shapes.TxNxA(0), symbolic_name='delta_1'),
            param('delta', Shapes.TxNxA(1), symbolic_name='delta_2'),
            param('delta', Shapes.TxNxA(2), symbolic_name='delta_3'),
            param('delta', Shapes.TxNxA(3), symbolic_name='delta_4'),
            param('delta', Shapes.TxNxA(4), symbolic_name='delta_5'),
            param('gamma', Shapes.TxNxA(0), symbolic_name='gamma_a'),
            param('gamma', Shapes.TxNxA(1), symbolic_name='gamma_b'),
            param('gamma', Shapes.TxNxA(2), symbolic_name='gamma_c'),
            param('rho', Shapes.TxNxA(0), symbolic_name='rho_1'),
            param('rho', Shapes.TxNxA(1), symbolic_name='rho_2'),
            param('rho', Shapes.TxNxA(2), symbolic_name='rho_3'),
            param('rho', Shapes.TxNxA(3), symbolic_name='rho_4'),
            param('rho', Shapes.TxNxA(4), symbolic_name='rho_5'),
        ])

    [S, E, Ia, Ip, Is, Ib, Ih, Ic1, Ic2, D, R] = symbols.compartment_symbols
    [beta_1, omega_1, omega_2, delta_1, delta_2, delta_3, delta_4, delta_5,
        gamma_a, gamma_b, gamma_c, rho_1, rho_2, rho_3, rho_4, rho_5] = symbols.attribute_symbols

    # formulate the divisor so as to avoid dividing by zero;
    # this is safe in this instance becase if the denominator is zero,
    # the numerator must also be zero
    N = Max(1, S + E + Ia + Ip + Is + Ib + Ih + Ic1 + Ic2 + R)
    lambda_1 = (omega_1 * Ia + Ip + Is + Ib + omega_2 * (Ih + Ic1 + Ic2)) / N

    return create_model(
        symbols=symbols,
        transitions=[
            edge(S, E, rate=beta_1 * lambda_1 * S),
            fork(
                edge(E, Ia, rate=E * delta_1 * rho_1),
                edge(E, Ip, rate=E * delta_1 * (1 - rho_1))
            ),
            edge(Ip, Is, rate=Ip * delta_2),
            fork(
                edge(Is, Ih, rate=Is * delta_3 * rho_2),
                edge(Is, Ic1, rate=Is * delta_3 * rho_3),
                edge(Is, Ib, rate=Is * delta_3 * (1 - rho_2 - rho_3))
            ),
            fork(
                edge(Ih, Ic1, rate=Ih * delta_4 * rho_4),
                edge(Ih, R, rate=Ih * delta_4 * (1 - rho_4))
            ),
            fork(
                edge(Ic1, D, rate=Ic1 * delta_5 * rho_5),
                edge(Ic1, Ic2, rate=Ic1 * delta_5 * (1 - rho_5))
            ),
            edge(Ia, R, rate=Ia * gamma_a),
            edge(Ib, R, rate=Ib * gamma_b),
            edge(Ic2, R, rate=Ic2 * gamma_c)
        ])
