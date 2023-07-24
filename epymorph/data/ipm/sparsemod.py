from sympy import Max

from epymorph.ipm.attribute import param
from epymorph.ipm.compartment_ipm import CompartmentModelIpmBuilder
from epymorph.ipm.compartment_model import (compartment, create_model,
                                            create_symbols, edge, fork)
from epymorph.ipm.ipm import IpmBuilder


def load() -> IpmBuilder:
    symbols = create_symbols(
        compartments=[
            compartment('S', 'susceptible'),
            compartment('E', 'exposed'),
            compartment('Ia', 'infected asymptomatic'),
            compartment('Ip', 'infected presymptomatic'),
            compartment('Is', 'infected symptomatic'),
            compartment('Ib', 'infected bed-rest'),
            compartment('Ih', 'infected hospitalized'),
            compartment('Ic1', 'infected in ICU'),
            compartment('Ic2', 'infected in ICU Step-Down'),
            compartment('D', 'deceased'),
            compartment('R', 'recovered')
        ],
        attributes=[
            param('beta_1', 'beta'),
            param('omega_1', 'omega', '0'),
            param('omega_2', 'omega', '1'),
            param('delta_1', 'delta', '0'),
            param('delta_2', 'delta', '1'),
            param('delta_3', 'delta', '2'),
            param('delta_4', 'delta', '3'),
            param('delta_5', 'delta', '4'),
            param('gamma_a', 'gamma', '0'),
            param('gamma_b', 'gamma', '1'),
            param('gamma_c', 'gamma', '2'),
            param('rho_1', 'rho', '0'),
            param('rho_2', 'rho', '1'),
            param('rho_3', 'rho', '2'),
            param('rho_4', 'rho', '3'),
            param('rho_5', 'rho', '4')
        ])

    [S, E, Ia, Ip, Is, Ib, Ih, Ic1, Ic2, D, R] = symbols.compartment_symbols
    [beta_1, omega_1, omega_2, delta_1, delta_2, delta_3, delta_4, delta_5,
        gamma_a, gamma_b, gamma_c, rho_1, rho_2, rho_3, rho_4, rho_5] = symbols.attribute_symbols

    # formulate the divisor so as to avoid dividing by zero;
    # this is safe in this instance becase if the denominator is zero,
    # the numerator must also be zero
    N = Max(1, S + E + Ia + Ip + Is + Ib + Ih + Ic1 + Ic2 + R)
    lambda_1 = (omega_1 * Ia + Ip + Is + Ib + omega_2 * (Ih + Ic1 + Ic2)) / N

    sparsemod = create_model(
        symbols=symbols,
        transitions=[
            edge(S, E, rate=beta_1 * lambda_1 * S),
            fork(
                edge(E, Ia, rate=E * delta_1 * rho_1),
                edge(E, Ip, rate=E * delta_1 * (1 - rho_1))
            ),
            edge(Ip, Is, rate=Ip * delta_2),
            fork(
                edge(Is, Ih,  rate=Is * delta_3 * rho_2),
                edge(Is, Ic1, rate=Is * delta_3 * rho_3),
                edge(Is, Ib,  rate=Is * delta_3 * (1 - rho_2 - rho_3))
            ),
            fork(
                edge(Ih, Ic1, rate=Ih * delta_4 * rho_4),
                edge(Ih, R,   rate=Ih * delta_4 * (1 - rho_4))
            ),
            fork(
                edge(Ic1, D,   rate=Ic1 * delta_5 * rho_5),
                edge(Ic1, Ic2, rate=Ic1 * delta_5 * (1 - rho_5))
            ),
            edge(Ia, R,  rate=Ia * gamma_a),
            edge(Ib, R,  rate=Ib * gamma_b),
            edge(Ic2, R, rate=Ic2 * gamma_c)
        ])

    return CompartmentModelIpmBuilder(sparsemod)
