from __future__ import annotations

import numpy as np

from epymorph.clock import Tick
from epymorph.context import SimContext
from epymorph.epi import (CompartmentalIpm, CompartmentModel, CompartmentSum,
                          EventId, IndependentEvent, Ipm, IpmBuilder,
                          SplitEvent, StateId, SubEvent)
from epymorph.util import Compartments
from epymorph.world import Location

ei = IndependentEvent
es = SplitEvent
esub = SubEvent


def load() -> IpmBuilder:
    return Builder()


class Builder(IpmBuilder):
    def __init__(self):
        super().__init__(num_compartments=11, num_events=14)

    def build(self, ctx: SimContext) -> Ipm:
        # States
        S = StateId("Susceptible")  # 0
        E = StateId("Exposed")  # 1
        Ia = StateId("Infected Asymptomatic")  # 2
        Ip = StateId("Infected Pre-symptomatic")  # 3
        Is = StateId("Infected Mild Symptoms")  # 4
        Ib = StateId("Infected Bed Rest")  # 5
        Ih = StateId("Infected Hospitalized")  # 6
        Ic1 = StateId("Infected ICU")  # 7
        Ic2 = StateId("Infected Step-down ICU")  # 8
        D = StateId("Deceased")  # 9
        R = StateId("Recovered")  # 10

        # Events
        S_E = EventId("Exposure")
        E_Ia = EventId("Infection (Asymptomatic)")
        E_Ip = EventId("Infection (Pre-symptomatic)")
        Ia_R = EventId("Recovery (Asymptomatic)")
        Ip_Is = EventId("Progression (Symptomatic)")
        Is_Ib = EventId("Progression (Bed Rest)")
        Is_Ih = EventId("Progression (Hospitalization)")
        Is_Ic1 = EventId("Progression (ICU from Symptomatic)")
        Ih_Ic1 = EventId("Progression (ICU from Hospital)")
        Ic1_Ic2 = EventId("Progression (Step-down ICU)")
        Ic1_D = EventId("Death")
        Ib_R = EventId("Recovery (Bed Rest)")
        Ic2_R = EventId("Recovery (Step-down ICU)")
        Ih_R = EventId("Recovery (Hospital)")

        # Params
        beta_1 = ctx.param['beta_1']
        omega_1 = ctx.param['omega_1']
        omega_2 = ctx.param['omega_2']
        delta_1 = ctx.param['delta_1']
        delta_2 = ctx.param['delta_2']
        delta_3 = ctx.param['delta_3']
        delta_4 = ctx.param['delta_4']
        delta_5 = ctx.param['delta_5']
        gamma_a = ctx.param['gamma_a']
        gamma_b = ctx.param['gamma_b']
        gamma_c = ctx.param['gamma_c']
        rho_1 = ctx.param['rho_1']
        rho_2 = ctx.param['rho_2']
        rho_3 = ctx.param['rho_3']
        rho_4 = ctx.param['rho_4']
        rho_5 = ctx.param['rho_5']

        # Rate expressions
        def expose(ctx: SimContext, tick: Tick, loc: Location, cs: CompartmentSum) -> int:
            lambda_1 = (omega_1 * cs[2] + cs[3] + cs[4] +
                        cs[5] + omega_2 * (cs[6] + cs[7] + cs[8])) / (cs.total - cs[9])
            return beta_1 * lambda_1 * cs[0]

        def e_progress(ctx: SimContext, tick: Tick, loc: Location, cs: CompartmentSum) -> int:
            return delta_1 * cs[1]

        def ia_recover(ctx: SimContext, tick: Tick, loc: Location, cs: CompartmentSum) -> int:
            return gamma_a * cs[2]

        def ib_recover(ctx: SimContext, tick: Tick, loc: Location, cs: CompartmentSum) -> int:
            return gamma_b * cs[5]

        def ic2_recover(ctx: SimContext, tick: Tick, loc: Location, cs: CompartmentSum) -> int:
            return gamma_c * cs[8]

        def ip_progress(ctx: SimContext, tick: Tick, loc: Location, cs: CompartmentSum) -> int:
            return delta_2 * cs[3]

        def is_progress(ctx: SimContext, tick: Tick, loc: Location, cs: CompartmentSum) -> int:
            return delta_3 * cs[4]

        def ih_progress(ctx: SimContext, tick: Tick, loc: Location, cs: CompartmentSum) -> int:
            return delta_4 * cs[6]

        def ic1_progress(ctx: SimContext, tick: Tick, loc: Location, cs: CompartmentSum) -> int:
            return delta_5 * cs[7]

        # Model
        model = CompartmentModel(
            states=[S, E, Ia, Ip, Is, Ib, Ih, Ic1, Ic2, D, R],
            events=[S_E, E_Ia, E_Ip, Ia_R, Ip_Is, Is_Ib, Is_Ih,
                    Is_Ic1, Ih_Ic1, Ic1_Ic2, Ic1_D, Ib_R, Ic2_R, Ih_R],
            transitions=[
                ei(S_E, S, E, expose),
                es(E, [
                    esub(E_Ia, Ia, rho_1),
                    esub(E_Ip, Ip, 1.0 - rho_1)
                ], e_progress),
                ei(Ia_R, Ia, R, ia_recover),
                ei(Ip_Is, Ip, Is, ip_progress),
                es(Is, [
                    esub(Is_Ib, Ib, 1.0 - rho_2 - rho_3),
                    esub(Is_Ih, Ih, rho_2),
                    esub(Is_Ic1, Ic1, rho_3)
                ], is_progress),
                ei(Ib_R, Ib, R, ib_recover),
                es(Ih, [
                    esub(Ih_Ic1, Ic1, rho_4),
                    esub(Ih_R, R, 1.0 - rho_4)
                ], ih_progress),
                es(Ic1, [
                    esub(Ic1_D, D, rho_5),
                    esub(Ic1_Ic2, Ic2, 1.0 - rho_5)
                ], ic1_progress),
                ei(Ic2_R, Ic2, R, ic2_recover)
            ])

        return CompartmentalIpm(ctx, model)

    def verify(self, ctx: SimContext) -> None:
        def g(name):
            if name not in ctx.geo:
                raise Exception(f"geo missing {name}")

        def p(name):
            if name not in ctx.param:
                raise Exception(f"params missing {name}")

        g("population")
        p("infection_seed_loc")
        p("infection_seed_size")
        p("beta_1")
        p("omega_1")
        p("omega_2")
        p("delta_1")
        p("delta_2")
        p("delta_3")
        p("delta_4")
        p("delta_5")
        p("gamma_a")
        p("gamma_b")
        p("gamma_c")
        p("rho_1")
        p("rho_2")
        p("rho_3")
        p("rho_4")
        p("rho_5")

    def initialize_compartments(self, ctx: SimContext) -> list[Compartments]:
        # Initial compartments based on population (C0)
        population = ctx.geo['population']
        # With a seeded infection (C1) in one location
        si = ctx.param['infection_seed_loc']
        sn = ctx.param['infection_seed_size']
        cs = np.zeros((ctx.nodes, ctx.compartments), dtype=np.int_)
        cs[:, 0] = population
        cs[si, 0] -= sn
        cs[si, 1] += sn
        return list(cs)
