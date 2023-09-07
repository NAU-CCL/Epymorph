from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

from epymorph.clock import Tick
from epymorph.context import Compartments, Events, SimContext, SimDType
from epymorph.ipm.attribute import (AttributeException, AttributeGetter,
                                    adapt_context, compile_getter)
from epymorph.ipm.compartment_model import (CompartmentModel, EdgeDef, ForkDef,
                                            TransitionDef)
from epymorph.ipm.ipm import Ipm, IpmBuilder
from epymorph.ipm.sympy_shim import (Symbol, SympyLambda, lambdify,
                                     lambdify_list)
from epymorph.movement.world import Location


@dataclass(frozen=True)
class IndependentTransition:
    """Represents a single edge. Effectively: `poisson(rate * tau)`"""
    rate_lambda: SympyLambda


@dataclass(frozen=True)
class ForkedTransition:
    """Represents a fork. Effectively: `multinomial(poisson(rate * tau), prob)`"""
    size: int
    rate_lambda: SympyLambda
    prob_lambda: SympyLambda


IpmTransition = IndependentTransition | ForkedTransition
"""A lambdified version of the compartment model transition, for use in the IPM."""


def compile_transition(transition: TransitionDef, rate_params: list[Symbol]) -> IpmTransition:
    """Compile a model transition for efficient evaluation within the IPM."""
    match transition:
        case EdgeDef(rate, _, _):
            rate_lambda = lambdify(rate_params, rate)
            return IndependentTransition(rate_lambda)
        case ForkDef(rate, edges, prob):
            size = len(edges)
            rate_lambda = lambdify(rate_params, rate)
            prob_lambda = lambdify_list(rate_params, prob)
            return ForkedTransition(size, rate_lambda, prob_lambda)


class CompartmentModelIpmBuilder(IpmBuilder):
    """The IpmBuilder for all IPMs driven by a CompartmentModel."""
    model: CompartmentModel

    def __init__(self, model: CompartmentModel):
        super().__init__(
            num_compartments=model.num_compartments,
            num_events=model.num_events
        )
        self.model = model

    def verify(self, ctx: SimContext) -> None:
        errors = []
        for attr in self.model.attributes:
            try:
                attr.verify(ctx)
            except AttributeException as e:
                errors.append(e)

        if len(errors) > 0:
            # TODO: better exception type for this
            raise Exception("IPM attribute requirements were not met. "
                            + "See errors:" + "".join(f"\n- {e}" for e in errors))

    def compartment_tags(self) -> list[list[str]]:
        return [c.tags for c in self.model.compartments]

    def build(self, ctx: SimContext) -> Ipm:
        rate_params = [
            *(c.symbol for c in self.model.compartments),
            *(a.symbol for a in self.model.attributes)
        ]

        adapted_ctx = adapt_context(ctx, self.model.attributes)

        return CompartmentModelIpm(
            adapted_ctx,
            self.model,
            attr_getters=[compile_getter(adapted_ctx, a)
                          for a in self.model.attributes],
            transitions=[compile_transition(t, rate_params)
                         for t in self.model.transitions])


class CompartmentModelIpm(Ipm):
    """The IPM instance for all IPMs driven by a CompartmentModel."""
    ctx: SimContext
    model: CompartmentModel
    attr_getters: list[AttributeGetter]
    transitions: list[IpmTransition]

    def __init__(self,
                 ctx: SimContext,
                 model: CompartmentModel,
                 attr_getters: list[AttributeGetter],
                 transitions: list[IpmTransition]):
        super().__init__(ctx)
        self.model = model
        self.attr_getters = attr_getters
        self.transitions = transitions

    def _rate_args(self, loc: Location, effective: Compartments, tick: Tick) -> list[Any]:
        """Assemble rate function arguments for this location/tick."""
        attribs = (f(loc, tick) for f in self.attr_getters)
        # NOTE: if SimDType is ever adjusted smaller than int64, you may want to do something like
        # return [*(effective.astype(np.int64)), *attribs]
        # We must be careful that whatever math is happening in the IPM won't overflow the numerical
        # representation being used.
        return [*effective, *attribs]

    def _eval_rates(self, rate_args: list[Any], tau: float) -> NDArray[SimDType]:
        """Evaluate the event rates and do random draws for all transition events."""
        occurrences = np.zeros(self.ctx.events, dtype=SimDType)
        index = 0
        for t in self.transitions:
            match t:
                case IndependentTransition(rate_lambda):
                    rate = rate_lambda(rate_args)
                    occurrences[index] = self.ctx.rng.poisson(rate * tau)
                    index += 1
                case ForkedTransition(size, rate_lambda, prob_lambda):
                    rate = rate_lambda(rate_args)
                    base = self.ctx.rng.poisson(rate * tau)
                    prob = prob_lambda(rate_args)
                    stop = index + size
                    occurrences[index:stop] = self.ctx.rng.multinomial(
                        base, prob)
                    index = stop
        return occurrences

    def events(self, loc: Location, tick: Tick) -> Events:
        # Get effective population for each compartment.
        effective = loc.get_compartments()

        # Calculate how many events we expect to happen this tick.
        rate_args = self._rate_args(loc, effective, tick)
        occur = self._eval_rates(rate_args, tick.tau)

        # Check for event overruns leaving each compartment and correct counts.
        for cidx, eidxs in enumerate(self.model.events_leaving_compartment):
            available = effective[cidx]
            n = len(eidxs)
            if n == 0:
                # Compartment has no outward edges; nothing to do here.
                continue
            elif n == 1:
                # Compartment only has one outward edge; just "min".
                eidx = eidxs[0]
                occur[eidx] = min(occur[eidx], available)
            elif n == 2:
                # Compartment has two outward edges:
                # use hypergeo to select which events "actually" happened.
                desired0, desired1 = occur[eidxs]
                if desired0 + desired1 > available:
                    drawn0 = self.ctx.rng.hypergeometric(
                        desired0, desired1, available)
                    occur[eidxs] = [drawn0, available - drawn0]
            else:
                # Compartment has more than two outwards edges:
                # use multivariate hypergeometric to select which events "actually" happened.
                desired = occur[eidxs]
                if np.sum(desired) > available:
                    occur[eidxs] = self.ctx.rng.multivariate_hypergeometric(
                        desired, available)
        return occur

    def _random_event_order(self) -> Iterable[int]:
        # this function exists to convince the type system that, yes, this is in fact a 1D array of ints
        return self.ctx.rng.permutation(self.ctx.events)

    def apply_events(self, loc: Location, es: Events) -> None:
        # For each event, redistribute across loc's pops.
        #
        # Process events in random order as a (probably somewhat naive) way to avoid biasing
        # events, since a fixed order risks "eating up" all the available individuals every round.
        #
        # The best option might be to shuffle a "deck" of event occurrences, draw an event, and then
        # draw a random individual (without replacement) to assign to that event. Repeat until all
        # events are distributed. However that sounds like a major performance hit if we're not careful
        # how to do it, so we're going with this for now.
        available = loc.get_cohorts()
        occurrences = np.zeros(
            (available.shape[0], self.ctx.events), dtype=SimDType)
        for eidx in self._random_event_order():
            occur: int = es[eidx]  # type: ignore
            cidx = self.model.source_compartment_for_event[eidx]
            selected = self.ctx.rng.multivariate_hypergeometric(
                available[:, cidx], occur).astype(SimDType)
            occurrences[:, eidx] = selected
            available[:, cidx] -= selected

        # Now that events are assigned to pops, update pop compartments using apply matrix.
        deltas = np.matmul(
            occurrences, self.model.apply_matrix, dtype=SimDType)
        loc.update_cohorts(deltas)
