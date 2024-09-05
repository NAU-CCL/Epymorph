"""
A RUME (Runnable Modeling Experiment) is a package containing the critical components of an epymorph experiment.
Particular simulation tasks may require more information, but will certainly not require less.
A GPM (Geo-Population Model) is a subset of this configuration, and it is possible to combine multiple GPMs
into one multi-strata RUME.
"""

import dataclasses
import textwrap
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from itertools import accumulate, pairwise, starmap
from typing import Callable, Mapping, NamedTuple, OrderedDict, Self, Sequence, final

import numpy as np
from numpy.typing import NDArray
from sympy import Symbol

from epymorph.compartment_model import (
    BaseCompartmentModel,
    CombinedCompartmentModel,
    CompartmentModel,
    MetaEdgeBuilder,
    MultistrataModelSymbols,
    TransitionDef,
)
from epymorph.data_shape import SimDimensions
from epymorph.data_type import dtype_str
from epymorph.database import AbsoluteName, ModuleNamePattern, NamePattern
from epymorph.geography.scope import GeoScope
from epymorph.initializer import Initializer
from epymorph.movement_model import MovementClause, MovementModel
from epymorph.params import ParamSymbol, ParamValue, simulation_symbols
from epymorph.simulation import (
    DEFAULT_STRATA,
    META_STRATA,
    AttributeDef,
    TickDelta,
    TickIndex,
    TimeFrame,
    gpm_strata,
)
from epymorph.util import are_unique, map_values

#######
# GPM #
#######


class Gpm:
    """
    A GPM (short for Geo-Population Model) combines an IPM, MM, and initialization scheme.
    Most often, a GPM is used to specify the modules to be used for one of the several population strata
    that make up a RUME.
    """

    name: str
    ipm: CompartmentModel
    mm: MovementModel
    init: Initializer
    params: Mapping[ModuleNamePattern, ParamValue]

    def __init__(
        self,
        name: str,
        ipm: CompartmentModel,
        mm: MovementModel,
        init: Initializer,
        params: Mapping[str, ParamValue] | None = None,
    ):
        self.name = name
        self.ipm = ipm
        self.mm = mm
        self.init = init
        self.params = {ModuleNamePattern.parse(k): v for k, v in (params or {}).items()}


########
# RUME #
########


GEO_LABELS = AbsoluteName(META_STRATA, "geo", "label")
"""
If this attribute is provided to a RUME, it will be used as labels for the geo node.
Otherwise we'll use the node IDs from the geo scope.
"""


class _CombineTauStepsResult(NamedTuple):
    new_tau_steps: tuple[float, ...]
    start_mapping: dict[str, dict[int, int]]
    stop_mapping: dict[str, dict[int, int]]


def combine_tau_steps(
    strata_tau_lengths: dict[str, Sequence[float]],
) -> _CombineTauStepsResult:
    """
    When combining movement models with different tau steps, it is necessary to create a
    new tau step scheme which can accomodate them all. This function performs that calculation,
    returning both the new tau steps (a list of tau lengths) and the mapping by strata from
    old tau step indices to new tau step indices, so that movement models can be adjusted as necessary.
    For example, if MM A has tau steps [1/3, 2/3] and MM B has tau steps [1/2, 1/2] -- the resulting
    combined tau steps are [1/3, 1/6, 1/2].
    """

    # Convert the tau lengths into the starting point and stopping point for each tau step.
    # Starts and stops are expressed as fractions of one day.
    def tau_starts(taus: Sequence[float]) -> Sequence[float]:
        return [0.0, *accumulate(taus)][:-1]

    def tau_stops(taus: Sequence[float]) -> Sequence[float]:
        return [*accumulate(taus)]

    strata_tau_starts = map_values(tau_starts, strata_tau_lengths)
    strata_tau_stops = map_values(tau_stops, strata_tau_lengths)

    # Now we combine all the tau starts set-wise, and sort.
    # These will be the tau steps for our combined simulation.
    combined_tau_starts = list({s for curr in strata_tau_starts.values() for s in curr})
    combined_tau_starts.sort()
    combined_tau_stops = list({s for curr in strata_tau_stops.values() for s in curr})
    combined_tau_stops.sort()

    # Now calculate the combined tau lengths.
    combined_tau_lengths = tuple(
        stop - start for start, stop in zip(combined_tau_starts, combined_tau_stops)
    )

    # But the individual strata MMs are indexed by their original tau steps,
    # so we need to calculate the appropriate re-indexing to the new tau steps
    # which will allow us to convert [strata MM tau index] -> [total sim tau index].
    tau_start_mapping = {
        name: {i: combined_tau_starts.index(x) for i, x in enumerate(curr)}
        for name, curr in strata_tau_starts.items()
    }
    tau_stop_mapping = {
        name: {i: combined_tau_stops.index(x) for i, x in enumerate(curr)}
        for name, curr in strata_tau_stops.items()
    }

    return _CombineTauStepsResult(
        combined_tau_lengths, tau_start_mapping, tau_stop_mapping
    )


def remap_taus(
    strata_mms: list[tuple[str, MovementModel]],
) -> OrderedDict[str, MovementModel]:
    """
    When combining movement models with different tau steps, it is necessary to create a
    new tau step scheme which can accomodate them all.
    """
    new_tau_steps, start_mapping, stop_mapping = combine_tau_steps(
        {strata: mm.steps for strata, mm in strata_mms}
    )

    def clause_remap_tau(clause: MovementClause, strata: str) -> MovementClause:
        leave_step = start_mapping[strata][clause.leaves.step]
        return_step = stop_mapping[strata][clause.returns.step]

        clone = deepcopy(clause)
        clone.leaves = TickIndex(leave_step)
        clone.returns = TickDelta(clause.returns.days, return_step)
        return clone

    def model_remap_tau(orig_model: MovementModel, strata: str) -> MovementModel:
        clone = deepcopy(orig_model)
        clone.steps = new_tau_steps
        clone.clauses = tuple(clause_remap_tau(c, strata) for c in orig_model.clauses)
        return clone

    return OrderedDict(
        [
            (strata_name, model_remap_tau(model, strata_name))
            for strata_name, model in strata_mms
        ]
    )


@dataclass(frozen=True)
class Rume(ABC):
    """
    A RUME (or Runnable Modeling Experiment) contains the configuration of an
    epymorph-style simulation. It brings together one or more IPMs, MMs, initialization routines,
    and a geo-temporal scope. Model parameters can also be specified on a RUME.
    The RUME will eventually be used to construct a Simulation, which is an
    algorithm that uses a RUME to produce some results -- in the most basic case,
    running a disease simulation and providing time-series results of the disease model.
    """

    strata: Sequence[Gpm]
    ipm: BaseCompartmentModel
    mms: OrderedDict[str, MovementModel]
    scope: GeoScope
    time_frame: TimeFrame
    params: Mapping[NamePattern, ParamValue]
    dim: SimDimensions = field(init=False)

    def __post_init__(self):
        if not are_unique(g.name for g in self.strata):
            msg = "Strata names must be unique; duplicate found."
            raise ValueError(msg)

        # We can get the tau step lengths from a movement model.
        # In a multistrata model, there will be multiple remapped MMs,
        # but they all have the same set of tau steps so it doesn't matter
        # which we use. (Using the first one is safe.)
        first_strata = self.strata[0].name
        tau_step_lengths = self.mms[first_strata].steps

        dim = SimDimensions.build(
            tau_step_lengths=tau_step_lengths,
            start_date=self.time_frame.start_date,
            days=self.time_frame.duration_days,
            nodes=len(self.scope.get_node_ids()),
            compartments=self.ipm.num_compartments,
            events=self.ipm.num_events,
        )
        object.__setattr__(self, "dim", dim)

    @cached_property
    def requirements(self) -> Mapping[AbsoluteName, AttributeDef]:
        """Returns the attributes required by the RUME."""

        def generate_items():
            # IPM attributes are already fully named.
            yield from self.ipm.requirements_dict.items()
            # Name the MM and Init attributes.
            for gpm in self.strata:
                strata_name = gpm_strata(gpm.name)
                for a in gpm.mm.requirements:
                    yield AbsoluteName(strata_name, "mm", a.name), a
                for a in gpm.init.requirements:
                    yield AbsoluteName(strata_name, "init", a.name), a

        return OrderedDict(generate_items())

    @cached_property
    def compartment_mask(self) -> Mapping[str, NDArray[np.bool_]]:
        """
        Masks that describe which compartments belong in the given strata.
        For example: if the model has three strata ('a', 'b', and 'c') with three compartments each,
        `strata_compartment_mask('b')` returns `[0 0 0 1 1 1 0 0 0]`
        (where 0 stands for False and 1 stands for True).
        """

        def mask(length: int, true_slice: slice) -> NDArray[np.bool_]:
            # A boolean array with the given slice set to True, all others False
            m = np.zeros(shape=length, dtype=np.bool_)
            m[true_slice] = True
            return m

        # num of compartments in the combined IPM
        C = self.ipm.num_compartments
        # num of compartments in each strata
        strata_cs = [gpm.ipm.num_compartments for gpm in self.strata]
        # start and stop index for each strata
        strata_ranges = pairwise([0, *accumulate(strata_cs)])
        # map stata name to the mask for each strata
        return dict(
            zip(
                [g.name for g in self.strata],
                [mask(C, s) for s in starmap(slice, strata_ranges)],
            )
        )

    @cached_property
    def compartment_mobility(self) -> Mapping[str, NDArray[np.bool_]]:
        """Masks that describe which compartments should be considered subject to movement in a particular strata."""
        # The mobility mask for all strata.
        all_mobility = np.array(
            ["immobile" not in c.tags for c in self.ipm.compartments], dtype=np.bool_
        )
        # Mobility for a single strata is all_mobility boolean-and whether the compartment is in that strata.
        return {
            strata.name: all_mobility & self.compartment_mask[strata.name]
            for strata in self.strata
        }

    @abstractmethod
    def name_display_formatter(self) -> Callable[[AbsoluteName | NamePattern], str]:
        """Returns a function for formatting attribute/parameter names."""

    def params_description(self) -> str:
        """Provide a description of all attributes required by the RUME."""
        format_name = self.name_display_formatter()
        lines = []
        for name, attr in self.requirements.items():
            properties = [
                f"type: {dtype_str(attr.type)}",
                f"shape: {attr.shape}",
            ]
            if attr.default_value is not None:
                properties.append(f"default: {attr.default_value}")
            lines.append(f"{format_name(name)} ({', '.join(properties)})")
            if attr.comment is not None:
                comment_lines = textwrap.wrap(
                    attr.comment,
                    initial_indent="    ",
                    subsequent_indent="    ",
                )
                lines.extend(comment_lines)
            lines.append("")
        return "\n".join(lines)

    def generate_params_dict(self) -> str:
        """Generate a skeleton dictionary you can use to provide parameter values to the room."""
        format_name = self.name_display_formatter()
        lines = ["{"]
        for name, attr in self.requirements.items():
            value = "PLACEHOLDER"
            if attr.default_value is not None:
                value = str(attr.default_value)
            lines.append(f'    "{format_name(name)}": {value},')
        lines.append("}")
        return "\n".join(lines)

    @staticmethod
    def symbols(*symbols: ParamSymbol) -> tuple[Symbol, ...]:
        """Convenient function to retrieve the symbols used to represent simulation quantities."""
        return simulation_symbols(*symbols)

    def with_time_frame(self, time_frame: TimeFrame) -> Self:
        """Create a RUME with a new time frame."""
        # TODO: do we need to go through all of the params and subset any that are time-based?
        # How would that work? Or maybe reconciling to time frame happens at param evaluation time...
        return dataclasses.replace(self, time_frame=time_frame)


@dataclass(frozen=True)
class SingleStrataRume(Rume):
    """A RUME with a single strata."""

    ipm: CompartmentModel

    @classmethod
    def build(
        cls,
        ipm: CompartmentModel,
        mm: MovementModel,
        init: Initializer,
        scope: GeoScope,
        time_frame: TimeFrame,
        params: Mapping[str, ParamValue],
    ) -> Self:
        """Create a RUME with only a single strata."""
        return cls(
            strata=[Gpm(DEFAULT_STRATA, ipm, mm, init, {})],
            ipm=ipm,
            mms=OrderedDict([(DEFAULT_STRATA, mm)]),
            scope=scope,
            time_frame=time_frame,
            params={NamePattern.parse(k): v for k, v in params.items()},
        )

    def name_display_formatter(self) -> Callable[[AbsoluteName | NamePattern], str]:
        """Returns a function for formatting attribute/parameter names."""
        return lambda n: f"{n.module}::{n.id}"


@dataclass(frozen=True)
class MultistrataRume(Rume):
    """A RUME with a multiple strata."""

    ipm: CombinedCompartmentModel

    @classmethod
    def build(
        cls,
        strata: Sequence[Gpm],
        meta_requirements: Sequence[AttributeDef],
        meta_edges: MetaEdgeBuilder,
        scope: GeoScope,
        time_frame: TimeFrame,
        params: Mapping[str, ParamValue],
    ) -> Self:
        """Create a multistrata RUME by combining one GPM per strata."""
        return cls(
            strata=strata,
            # Combine IPMs
            ipm=CombinedCompartmentModel(
                strata=[(gpm.name, gpm.ipm) for gpm in strata],
                meta_requirements=meta_requirements,
                meta_edges=meta_edges,
            ),
            # Combine MMs
            mms=remap_taus([(gpm.name, gpm.mm) for gpm in strata]),
            scope=scope,
            time_frame=time_frame,
            params={NamePattern.parse(k): v for k, v in params.items()},
        )

    def name_display_formatter(self) -> Callable[[AbsoluteName | NamePattern], str]:
        """Returns a function for formatting attribute/parameter names."""
        return str


class MultistrataRumeBuilder(ABC):
    """Create a multi-strata RUME by combining GPMs, one for each strata."""

    strata: Sequence[Gpm]
    """The strata that are part of this RUME."""

    meta_requirements: Sequence[AttributeDef]
    """
    A set of additional requirements which are needed by the meta-edges
    in our combined compartment model.
    """

    @abstractmethod
    def meta_edges(self, symbols: MultistrataModelSymbols) -> list[TransitionDef]:
        """
        When implementing a MultistrataRumeBuilder, override this method
        to build the meta-transition-edges -- the edges which represent
        cross-strata interactions. You are given a reference to this model's symbols library
        so you can build expressions for the transition rates.
        """

    @final
    def build(
        self,
        scope: GeoScope,
        time_frame: TimeFrame,
        params: Mapping[str, ParamValue],
    ) -> MultistrataRume:
        """Build the RUME."""
        return MultistrataRume.build(
            self.strata,
            self.meta_requirements,
            self.meta_edges,
            scope,
            time_frame,
            params,
        )
