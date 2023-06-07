from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from epymorph.clock import Tick
from epymorph.context import SimContext
from epymorph.util import Compartments, Events
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


@dataclass(frozen=True)
class StateId:
    id: str


@dataclass(frozen=True)
class EventId:
    id: str


@dataclass(frozen=True)
class IndependentEvent:
    id: EventId
    state_from: StateId
    state_to: StateId
    rate: Callable[[SimContext, Tick, Location], int]


@dataclass(frozen=True)
class SplitEvent:
    state_from: StateId
    rate: Callable[[SimContext, Tick, Location], int]
    states_to: list[StateId]
    eids: list[EventId]
    probs: list[float]

    def __post_init__(self):
        # TODO: error messages
        assert len(self.states_to) > 0
        assert len(self.states_to) == len(self.eids)
        assert len(self.states_to) == len(self.probs)
        assert sum(self.probs) == 1


Event = IndependentEvent | SplitEvent


class IpmModel:
    states: list[StateId]
    events: list[EventId]
    transitions: list[Event]

    def __init__(self, states: list[StateId], events: list[EventId], transitions: list[Event]):
        self.states = states
        self.events = events
        self.transitions = transitions

        # Checks:
        # 1. the given states should match the set discovered in transitions
        # 2. the given events should match the set discovered in transitions
        # 3. each event should only be represented once in transitions
        state_set = set[StateId]()
        event_set = set[EventId]()
        for t in transitions:
            match t:
                case IndependentEvent(eid, sid1, sid2, _):
                    state_set.add(sid1)
                    state_set.add(sid2)
                    assert eid not in event_set  # TODO: error message
                    event_set.add(eid)
                case SplitEvent(sid1, _, states_to, eids, _):
                    state_set.add(sid1)
                    state_set.update(states_to)
                    assert len(event_set.intersection(eids)
                               ) == 0  # TODO: error message
                    event_set.update(eids)
        # TODO: error messages
        assert state_set == set(states)
        assert event_set == set(events)

    def to_apply_matrix(self):
        e = len(self.events)
        s = len(self.states)
        mat = np.zeros((e, s), dtype=np.int_)
        for t in self.transitions:
            match t:
                case IndependentEvent(eid, sid1, sid2, _):
                    sidx1 = self.states.index(sid1)
                    sidx2 = self.states.index(sid2)
                    eidx = self.events.index(eid)
                    mat[eidx, sidx1] = -1
                    mat[eidx, sidx2] = +1
                case SplitEvent(sid1, _, states_to, eids, _):
                    sidx1 = self.states.index(sid1)
                    for eid, sid2 in zip(eids, states_to):
                        sidx2 = self.states.index(sid2)
                        eidx = self.events.index(eid)
                        mat[eidx, sidx1] = -1
                        mat[eidx, sidx2] = +1
        return mat


class IpmBuilderFromModel(IpmBuilder):
    model: IpmModel

    def __init__(self, model: IpmModel):
        self.model = model
        super().__init__(len(model.states), len(model.events))

    def initialize_compartments(self, ctx: SimContext) -> list[Compartments]:
        # WARNING: makes a lot of assumptions that don't apply generally!
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
        return IpmForModel(ctx, self.model)


class IpmForModel(Ipm):
    ctx: SimContext
    model: IpmModel
    apply_matrix: NDArray[np.int_]

    # mapping from compartment index to the list of events which source from that compartment
    source_events: list[list[int]]
    # mapping from event to the compartment it sources from
    event_source: list[int]

    def __init__(self, ctx: SimContext, model: IpmModel):
        self.ctx = ctx
        self.model = model
        self.apply_matrix = model.to_apply_matrix()

        src = [list[int]() for _ in range(len(model.states))]
        evt = [0 for _ in range(len(model.events))]
        for t in model.transitions:
            match t:
                case IndependentEvent(eid, state_from, _, _):
                    sidx = model.states.index(state_from)
                    eidx = model.events.index(eid)
                    src[sidx].append(eidx)
                    evt[eidx] = sidx
                case SplitEvent(state_from, _, _, eids, _):
                    sidx = model.states.index(state_from)
                    eidxs = map(lambda x: model.events.index(x), eids)
                    src[sidx].extend(eidxs)
                    for e in eidxs:
                        evt[e] = sidx
        self.source_events = src
        self.event_source = evt

    def events(self, loc: Location, tick: Tick) -> Events:
        events = np.zeros(self.ctx.events, dtype=int)
        for t in self.model.transitions:
            match t:
                case IndependentEvent(eid, state_from, state_to, rate_expr):
                    rate = rate_expr(self.ctx, tick, loc)
                    occur = self.ctx.rng.poisson(rate * tick.tau)
                    events[self.model.events.index(eid)] = occur
                case SplitEvent(state_from, rate_expr, states_to, eids, probs):
                    rate = rate_expr(self.ctx, tick, loc)
                    occur = self.ctx.rng.poisson(rate * tick.tau)
                    splits = self.ctx.rng.multinomial(occur, probs)
                    for e, n in zip(eids, splits):
                        events[self.model.events.index(e)] = n

        # Check for event overruns and reduce counts.
        individuals = loc.compartment_totals
        for sidx, eidxs in enumerate(self.source_events):
            if len(eidxs) == 0:
                continue
            elif len(eidxs) == 1:
                eidx = eidxs[0]
                desired = events[eidx]
                events[eidx] = min(events[eidx], individuals[sidx])
            else:
                desired = events[eidxs]
                available = individuals[sidx]
                if np.sum(desired) > available:
                    events[eidxs] = self.ctx.rng.multivariate_hypergeometric(
                        desired, available)
        return events

    def apply_events(self, loc: Location, es: Events) -> None:
        cs = np.array([pop.compartments for pop in loc.pops])
        es_by_pop = np.zeros((self.ctx.nodes, self.ctx.events), dtype=int)
        for eidx, occur in enumerate(es):
            sidx = self.event_source[eidx]
            es_by_pop[:, eidx] = self.ctx.rng.multivariate_hypergeometric(
                cs[:, sidx], occur)
        for pidx, pop in enumerate(loc.pops):
            deltas = np.matmul(es_by_pop[pidx], self.apply_matrix)
            pop.compartments += deltas
