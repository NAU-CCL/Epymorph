from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Generator

import numpy as np
from numpy.typing import NDArray

from epymorph.clock import Tick
from epymorph.context import SimContext
from epymorph.util import (Compartments, Events, NotUniqueException,
                           as_unique_set)
from epymorph.world import Location


class IpmBuilder(ABC):
    compartments: int
    events: int

    def __init__(self, num_compartments: int, num_events: int):
        self.compartments = num_compartments
        self.events = num_events

    def compartment_array(self) -> Compartments:
        """Build an empty compartment array of an appropriate size."""
        return np.zeros(self.compartments, dtype=np.int_)

    def event_array(self) -> Events:
        """Build an empty events array of an appropriate size."""
        return np.zeros(self.events, dtype=np.int_)

    @abstractmethod
    def verify(self, ctx: SimContext) -> None:
        """Verify whether or not this IPM has access to the data it needs."""
        pass

    @abstractmethod
    def initialize_compartments(self, ctx: SimContext) -> list[Compartments]:
        pass

    @abstractmethod
    def build(self, ctx: SimContext) -> Ipm:
        pass


class Ipm(ABC):
    """Superclass for an Intra-population Model,
    used to calculate transition events at each tau step."""
    ctx: SimContext

    def __init__(self, ctx: SimContext):
        self.ctx = ctx

    @abstractmethod
    def events(self, loc: Location,  tick: Tick) -> Events:
        """Calculate the events which took place in this tau step at the given location."""
        pass

    @abstractmethod
    def apply_events(self, loc: Location, es: Events) -> None:
        """Distribute events `es` among all populations at this location. (Modifies `loc`.)"""
        pass


############################################################
# Compartmental Model Object Model
############################################################


class InvalidModelException(Exception):
    pass


@dataclass(frozen=True)
class StateId:
    id: str


@dataclass(frozen=True)
class EventId:
    id: str


@dataclass(frozen=True)
class CompartmentSum:
    """Convenience class for calculating the compartment totals across multiple sub-populations."""
    compartments: Compartments
    total: int

    @classmethod
    def for_location(cls, loc: Location) -> CompartmentSum:
        cs = np.sum([p.compartments for p in loc.pops], axis=0)
        return cls(cs, cs.sum(dtype=np.int_))

    def __getitem__(self, i: int) -> np.int_:
        return self.compartments.__getitem__(i)


RateExpression = Callable[[SimContext, Tick, Location, CompartmentSum], int]


@dataclass(frozen=True)
class IndependentEvent:
    """
    An event from one state to another whose draw is completely independent from any other.
    It will be resolved as a poisson draw.
    """
    eid: EventId
    state_from: StateId
    state_to: StateId
    rate: RateExpression


@dataclass(frozen=True)
class SubEvent:
    """Part of a SplitEvent."""
    eid: EventId
    state_to: StateId
    prob: float


@dataclass(frozen=True)
class SplitEvent:
    """
    A set of events which leave the same state whose draws are co-dependent.
    Generally there is a rate at which all of the events happen and then there
    are probabilities that split this psuedo-event into two or more branches.
    This is resolved by doing a poisson draw followed by a multinomial.
    """
    state_from: StateId
    sub_events: list[SubEvent]
    rate: RateExpression
    probs: list[float] = field(init=False)
    """For convenience: the probabilities of each sub-event."""

    def __post_init__(self):
        # using setattr bypasses restriction on setting fields on frozen dataclasses
        object.__setattr__(self, 'probs', [e.prob for e in self.sub_events])
        if len(self.sub_events) < 2:
            raise InvalidModelException(
                "IPM SplitEvent must define at least two sub-events.")
        if sum(self.probs[:-1]) >= 1.0:
            # Note: when it comes to probability lists, numpy asserts that all elements
            # except the last sum to a value less than one and assumes the last probability
            # will receive the remainder (whatever that is).
            # Basically the last value is ignored, which is a questionable design philosophy.
            # e.g., if you gave it [0.75, 0.75], the *actual* values used internally are [0.75, 0.25]
            # and the same goes for [0.75, 0.01]
            # Purely for expedience, we've duplicated that behavior here.
            # But maybe we should actually normalize to 1, or make stricter assertions.
            raise InvalidModelException(
                "IPM SplitEvent's sub-event probabilities must be numpy compatible (sum to 1.0, kinda)")


Event = IndependentEvent | SplitEvent


def edges(events: list[Event]) -> Generator[tuple[(EventId, StateId, StateId)], None, None]:
    for e in events:
        match e:
            case IndependentEvent(eid, state_from, state_to, _):
                yield (eid, state_from, state_to)
            case SplitEvent(state_from, sub_events, _):
                for e in sub_events:
                    yield (e.eid, state_from, e.state_to)


class CompartmentModel:
    """A compartmental model of a disease process."""
    states: list[StateId]
    events: list[EventId]
    transitions: list[Event]

    def __init__(self, states: list[StateId], events: list[EventId], transitions: list[Event]):
        self.states = states
        self.events = events
        self.transitions = transitions

        # Checks:
        try:
            state_set = set(sid
                            for _, sid1, sid2 in edges(transitions)
                            for sid in [sid1, sid2])
            event_set = as_unique_set(
                list(eid for eid, _, _ in edges(transitions)))
            # 1. the given states should match the set discovered in transitions
            if state_set != set(states):
                raise InvalidModelException(
                    "IPM definition mismatch between states and transitions.")
            # 2. the given events should match the set discovered in transitions
            if event_set != set(events):
                raise InvalidModelException(
                    "IPM definition mismatch between events and transitions.")
        except NotUniqueException:
            # 3. each event should only be represented once in transitions
            raise InvalidModelException(
                "IPM defines multiple edges per event.")

    def to_apply_matrix(self):
        mat = np.zeros((len(self.events), len(self.states)), dtype=np.int_)
        for eid, sid1, sid2 in edges(self.transitions):
            sidx1 = self.states.index(sid1)
            sidx2 = self.states.index(sid2)
            eidx = self.events.index(eid)
            mat[eidx, sidx1] = -1
            mat[eidx, sidx2] = +1
        return mat

############################################################
# IPM for CompartmentModel
############################################################


class CompartmentalIpm(Ipm):
    """An IPM which knows how to evaluate a compartmental model."""
    ctx: SimContext
    model: CompartmentModel

    # a matrix defining how each event impacts each compartment (subtracting or adding individuals)
    _apply_matrix: NDArray[np.int_]
    # mapping from compartment index to the list of event indices which source from that compartment
    _source_events: list[list[int]]
    # mapping from event index to the compartment index it sources from
    _event_source: list[int]

    def __init__(self, ctx: SimContext, model: CompartmentModel):
        if ctx.compartments != len(model.states) or ctx.events != len(model.events):
            raise InvalidModelException(
                "SimContext and CompartmentModel are incompatible somehow.")
        self.ctx = ctx
        self.model = model
        self._apply_matrix = model.to_apply_matrix()
        # compute which events come from which source, and vice versa
        src = [list[int]() for _ in range(len(model.states))]
        evt = [0 for _ in range(len(model.events))]
        for eid, state_from, _ in edges(model.transitions):
            sidx = model.states.index(state_from)
            eidx = model.events.index(eid)
            src[sidx].append(eidx)
            evt[eidx] = sidx
        self._source_events = src
        self._event_source = evt

    def events(self, loc: Location, tick: Tick) -> Events:
        # First calculate how many events we expect to happen this tick.
        csum = CompartmentSum.for_location(loc)
        events = np.zeros(self.ctx.events, dtype=int)
        for t in self.model.transitions:
            match t:
                case IndependentEvent(eid, _, _, rate_expr):
                    rate = rate_expr(self.ctx, tick, loc, csum)
                    occur = self.ctx.rng.poisson(rate * tick.tau)
                    events[self.model.events.index(eid)] = occur
                case SplitEvent(_, sub_events, rate_expr):
                    rate = rate_expr(self.ctx, tick, loc, csum)
                    occur = self.ctx.rng.poisson(rate * tick.tau)
                    splits = self.ctx.rng.multinomial(occur, t.probs)
                    for e, n in zip(sub_events, splits):
                        events[self.model.events.index(e.eid)] = n

        # Check for event overruns leaving each compartment and reduce counts.
        individuals = loc.compartment_totals
        for sidx, eidxs in enumerate(self._source_events):
            n = len(eidxs)
            if n == 0:
                # Compartment has no outward edges; nothing to do here.
                continue
            elif n == 1:
                # Compartment only has one outward edge; just "min".
                eidx = eidxs[0]
                events[eidx] = min(events[eidx], individuals[sidx])
            elif n == 2:
                # Compartment has two outward edges:
                # use hypergeo to select which events "actually" happened.
                eidx0 = eidxs[0]
                eidx1 = eidxs[1]
                desired0 = events[eidx0]
                desired1 = events[eidx1]
                available = individuals[sidx]
                if desired0 + desired1 > available:
                    drawn0 = self.ctx.rng.hypergeometric(
                        desired0, desired1, available)
                    events[eidx0] = drawn0
                    events[eidx1] = available - drawn0
            else:
                # Compartment has more than two outwards edges:
                # use multivariate hypergeometric to select which events "actually" happened.
                desired = events[eidxs]
                available = individuals[sidx]
                if np.sum(desired) > available:
                    events[eidxs] = self.ctx.rng.multivariate_hypergeometric(
                        desired, available)
        return events

    def apply_events(self, loc: Location, es: Events) -> None:
        cs = np.array([pop.compartments for pop in loc.pops])
        es_by_pop = np.zeros((self.ctx.nodes, self.ctx.events), dtype=int)
        # For each event, redistribute across loc's pops
        for eidx, occur in enumerate(es):
            sidx = self._event_source[eidx]
            es_by_pop[:, eidx] = self.ctx.rng.multivariate_hypergeometric(
                cs[:, sidx], occur)
        # Now that events are assigned to pops, update pop compartments using apply matrix.
        for pidx, pop in enumerate(loc.pops):
            deltas = np.matmul(es_by_pop[pidx], self._apply_matrix)
            pop.compartments += deltas
