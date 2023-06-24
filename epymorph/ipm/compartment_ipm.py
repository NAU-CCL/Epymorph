from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import sympy
from numpy.typing import NDArray
from sympy import Symbol

from epymorph.clock import Tick
from epymorph.context import SimContext
from epymorph.epi import Ipm, IpmBuilder
from epymorph.ipm.attribute import (AttributeGetter, compile_getter,
                                    verify_attribute)
from epymorph.ipm.compartment_model import (CompartmentModel, EdgeDef, ForkDef,
                                            TransitionDef)
from epymorph.util import Compartments, Events, list_not_none
from epymorph.world import Location


@dataclass(frozen=True)
class IndependentTransition:
    rate_lambda: Any


@dataclass(frozen=True)
class ForkedTransition:
    size: int
    rate_lambda: Any
    prob_lambda: Any


IpmTransition = IndependentTransition | ForkedTransition


def compile_transition(transition: TransitionDef, rate_args: list[Symbol]) -> IpmTransition:
    match transition:
        case EdgeDef(_, _, rate):
            rate_lambda = sympy.lambdify([rate_args], rate)
            return IndependentTransition(rate_lambda)
        case ForkDef(rate, edges, prob):
            size = len(edges)
            rate_lambda = sympy.lambdify([rate_args], rate)
            prob_lambda = sympy.lambdify([rate_args], prob)
            return ForkedTransition(size, rate_lambda, prob_lambda)


class CompartmentModelIpmBuilder(IpmBuilder):
    model: CompartmentModel

    def __init__(self, model: CompartmentModel):
        self.model = model
        super().__init__(
            num_compartments=len(model.compartments),
            # TODO: maybe there's a better way to get num_events
            num_events=len(model.source_compartment_for_event)
        )

    def verify(self, ctx: SimContext) -> None:
        errors = list_not_none(verify_attribute(ctx, a)
                               for a in self.model.attributes)
        if len(errors) > 0:
            # TODO: better exception type for this
            raise Exception("IPM attribute requirements were not met. "
                            + "See errors:" + "".join(f"\n- {e}" for e in errors))

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

    def build(self, ctx: SimContext) -> Ipm:
        rate_args = [
            *(c.symbol for c in self.model.compartments),
            *(a.symbol for a in self.model.attributes)
        ]
        return CompartmentModelIpm(
            ctx,
            self.model,
            attr_getters=[compile_getter(ctx, a)
                          for a in self.model.attributes],
            transitions=[compile_transition(t, rate_args)
                         for t in self.model.transitions])


class CompartmentModelIpm(Ipm):
    ctx: SimContext
    model: CompartmentModel
    attr_getters: list[AttributeGetter]
    transitions: list[IpmTransition]

    def __init__(self, ctx: SimContext, model: CompartmentModel, attr_getters: list[AttributeGetter], transitions: list[IpmTransition]):
        self.ctx = ctx
        self.model = model
        self.attr_getters = attr_getters
        self.transitions = transitions

    def _rate_args(self, loc: Location, effective: Compartments, tick: Tick) -> list[Any]:
        """Assemble rate function arguments for this location/tick."""
        attribs = (f(loc, tick) for f in self.attr_getters)
        return [*effective, *attribs]

    def _eval_rates(self, params: list[Any]) -> NDArray[np.int_]:
        """Evaluate the event rates and do random draws for all transition events."""
        occurrences = np.zeros(self.ctx.events, dtype=int)
        index = 0
        for t in self.transitions:
            match t:
                case IndependentTransition(rate_lambda):
                    rate = rate_lambda(params)
                    occurrences[index] = self.ctx.rng.poisson(rate)
                    index += 1
                case ForkedTransition(size, rate_lambda, prob_lambda):
                    rate = rate_lambda(params)
                    base = self.ctx.rng.poisson(rate)
                    prob = prob_lambda(params)
                    stop = index + size
                    occurrences[index:stop] = self.ctx.rng.multinomial(
                        base, prob)
                    index = stop
        return occurrences

    def events(self, loc: Location, tick: Tick) -> Events:
        # Get effective population for each copmartment.
        all_pops = np.array([p.compartments for p in loc.pops], dtype=int)
        effective = np.sum(all_pops, axis=0)

        # Calculate how many events we expect to happen this tick.
        rate_args = self._rate_args(loc, effective, tick)
        erates = self._eval_rates(rate_args)
        expect = self.ctx.rng.poisson(erates * tick.tau)
        actual = expect.copy()

        # Check for event overruns leaving each compartment and reduce counts.
        for cidx, eidxs in enumerate(self.model.events_leaving_compartment):
            available = effective[cidx]
            n = len(eidxs)
            if n == 0:
                # Compartment has no outward edges; nothing to do here.
                continue
            elif n == 1:
                # Compartment only has one outward edge; just "min".
                eidx = eidxs[0]
                actual[eidx] = min(expect[eidx], available)
            elif n == 2:
                # Compartment has two outward edges:
                # use hypergeo to select which events "actually" happened.
                eidx0 = eidxs[0]
                eidx1 = eidxs[1]
                desired0 = expect[eidx0]
                desired1 = expect[eidx1]
                if desired0 + desired1 > available:
                    drawn0 = self.ctx.rng.hypergeometric(
                        desired0, desired1, available)
                    actual[eidx0] = drawn0
                    actual[eidx1] = available - drawn0
            else:
                # Compartment has more than two outwards edges:
                # use multivariate hypergeometric to select which events "actually" happened.
                desired = expect[eidxs]
                if np.sum(desired) > available:
                    actual[eidxs] = self.ctx.rng.multivariate_hypergeometric(
                        desired, available)
        return actual

    def apply_events(self, loc: Location, es: Events) -> None:
        # For each event, redistribute across loc's pops
        compartments = np.array([pop.compartments for pop in loc.pops])
        occurrences_by_pop = np.zeros(
            (self.ctx.nodes, self.ctx.events), dtype=int)
        for eidx, occur in enumerate(es):
            cidx = self.model.source_compartment_for_event[eidx]
            occurrences_by_pop[:, eidx] = self.ctx.rng.multivariate_hypergeometric(
                compartments[:, cidx], occur)

        # Now that events are assigned to pops, update pop compartments using apply matrix.
        for pidx, pop in enumerate(loc.pops):
            deltas = np.matmul(
                occurrences_by_pop[pidx], self.model.apply_matrix)
            pop.compartments += deltas
