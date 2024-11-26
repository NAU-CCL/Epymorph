"""
A RUME (Runnable Modeling Experiment) is a package containing the critical components
of an epymorph experiment. Particular simulation tasks may require more information,
but will certainly not require less. A GPM (Geo-Population Model) is a subset of this
configuration, and it is possible to combine multiple GPMs into one multi-strata RUME.
"""

import dataclasses
import textwrap
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from itertools import accumulate, pairwise, starmap
from typing import (
    Callable,
    Generic,
    Mapping,
    NamedTuple,
    OrderedDict,
    Self,
    Sequence,
    TypeVar,
    final,
)

import numpy as np
from numpy.typing import NDArray
from sympy import Symbol

from epymorph.adrio.adrio import Adrio
from epymorph.cache import CACHE_PATH
from epymorph.compartment_model import (
    BaseCompartmentModel,
    CombinedCompartmentModel,
    CompartmentModel,
    MetaEdgeBuilder,
    MultistrataModelSymbols,
    TransitionDef,
)
from epymorph.data_shape import Shapes, SimDimensions
from epymorph.data_type import SimArray, dtype_str
from epymorph.data_usage import estimate_report
from epymorph.database import (
    AbsoluteName,
    Database,
    DatabaseWithFallback,
    DatabaseWithStrataFallback,
    DataResolver,
    ModuleNamePattern,
    ModuleNamespace,
    NamePattern,
    ReqTree,
)
from epymorph.error import InitException
from epymorph.geography.scope import GeoScope
from epymorph.initializer import Initializer
from epymorph.movement_model import MovementClause, MovementModel
from epymorph.params import ParamSymbol, simulation_symbols
from epymorph.simulation import (
    DEFAULT_STRATA,
    META_STRATA,
    AttributeDef,
    ParamValue,
    TickDelta,
    TickIndex,
    gpm_strata,
)
from epymorph.time import TimeFrame
from epymorph.util import (
    KeyValue,
    are_unique,
    map_values,
)

#######
# GPM #
#######


@dataclass(frozen=True)
class Gpm:
    """
    A GPM (short for Geo-Population Model) combines an IPM, MM, and
    initialization scheme. Most often, a GPM is used to specify the modules
    to be used for one of the several population strata that make up a RUME.
    """

    name: str
    ipm: CompartmentModel
    mm: MovementModel
    init: Initializer
    params: Mapping[ModuleNamePattern, ParamValue] | None = field(default=None)

    # NOTE: constructing a ModuleNamePattern object is a bit awkward from an interface
    # perspective; much more ergonomic to just be able to use strings -- but that
    # requires a parsing call. Doing that parsing here is awkward for a dataclass.
    # And we could design around that but I'm not certain this feature isn't destinated
    # to be removed anyway... so for now users will have to do the parsing or maybe
    # we'll add a utility function that effectively does this:
    # params = {ModuleNamePattern.parse(k): v for k, v in (params or {}).items()}


########
# RUME #
########


GEO_LABELS = KeyValue(
    AbsoluteName(META_STRATA, "geo", "label"),
    AttributeDef(
        "label",
        str,
        Shapes.N,
        comment="Labels to use for each geo node.",
    ),
)
"""
If this attribute is provided to a RUME, it will be used as labels for the geo node.
Otherwise we'll use the labels from the geo scope.
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
    new tau step scheme which can accomodate them all. This function performs that
    calculation, returning both the new tau steps (a list of tau lengths) and the
    mapping by strata from old tau step indices to new tau step indices, so that
    movement models can be adjusted as necessary. For example, if MM A has
    tau steps [1/3, 2/3] and MM B has tau steps [1/2, 1/2] -- the resulting
    combined tau steps are [1/3, 1/6, 1/2].
    """

    # Convert the tau lengths into the starting point and stopping point for each
    # tau step.
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


GeoScopeT = TypeVar("GeoScopeT", bound=GeoScope)
GeoScopeT_co = TypeVar("GeoScopeT_co", covariant=True, bound=GeoScope)


@dataclass(frozen=True)
class Rume(ABC, Generic[GeoScopeT_co]):
    """
    A RUME (or Runnable Modeling Experiment) contains the configuration of an
    epymorph-style simulation. It brings together one or more IPMs, MMs, initialization
    routines, and a geo-temporal scope. Model parameters can also be specified.
    The RUME will eventually be used to construct a Simulation, which is an
    algorithm that uses a RUME to produce some results -- in the most basic case,
    running a disease simulation and providing time-series results of the disease model.
    """

    strata: Sequence[Gpm]
    ipm: BaseCompartmentModel
    mms: OrderedDict[str, MovementModel]
    scope: GeoScopeT_co
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
            nodes=len(self.scope.node_ids),
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

    def _params_database(
        self,
        override_params: Mapping[NamePattern, ParamValue]
        | Mapping[str, ParamValue]
        | None = None,
    ) -> Database[ParamValue]:
        label_name, _ = GEO_LABELS
        params_db = DatabaseWithStrataFallback(
            # RUME params are high priority,
            data={**self.params},
            # with fall back to strata params.
            children={
                **{
                    gpm_strata(gpm.name): Database(
                        {
                            k.to_absolute(gpm_strata(gpm.name)): v
                            for k, v in (gpm.params or {}).items()
                        }
                    )
                    for gpm in self.strata
                },
                # And provide a low-priority default for node labels.
                "meta": Database({label_name.to_pattern(): self.scope.labels}),
            },
        )
        # If override_params is not empty, wrap in another layer
        # where override_params have the highest priority.
        if override_params is not None and len(override_params) > 0:
            params_db = DatabaseWithFallback(
                {NamePattern.of(k): v for k, v in override_params.items()},
                params_db,
            )
        return params_db

    @cached_property
    def compartment_mask(self) -> Mapping[str, NDArray[np.bool_]]:
        """
        Masks that describe which compartments belong in the given strata.
        For example: if the model has three strata ('a', 'b', and 'c') with
        three compartments each,
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
        """
        Masks that describe which compartments should be considered
        subject to movement in a particular strata.
        """
        # The mobility mask for all strata.
        all_mobility = np.array(
            ["immobile" not in c.tags for c in self.ipm.compartments], dtype=np.bool_
        )
        # Mobility for a single strata is
        # all_mobility boolean-and whether the compartment is in that strata.
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
        """
        Generate a skeleton dictionary you can use to provide parameter values
        to the RUME.
        """
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
        """
        Convenient function to retrieve the symbols used to represent
        simulation quantities.
        """
        return simulation_symbols(*symbols)

    def with_time_frame(self, time_frame: TimeFrame) -> Self:
        """Create a RUME with a new time frame."""
        # TODO: do we need to go through all of the params and subset any
        # that are time-based?
        # How would that work? Or maybe reconciling to time frame happens
        # at param evaluation time...
        return dataclasses.replace(self, time_frame=time_frame)

    def estimate_data(
        self,
        *,
        max_bandwidth: int = 1000**2,  # default: 1 MB/s
    ) -> None:
        """Prints a report estimating the data requirements of this RUME.

        Includes data which must be downloaded and how much will be added to the file
        cache. Provides a projected download time based on the given assumed maximum
        network bandwidth (defaults to 1 MB/s).
        """

        estimates = [
            p.with_context_internal(scope=self.scope, dim=self.dim).estimate_data()
            for p in self.params.values()
            if isinstance(p, Adrio)
        ]

        lines = list[str]()
        if len(estimates) == 0:
            lines.append("ADRIO data usage is either negligible or non-estimable.")
        else:
            lines.append("ADRIO data usage estimation:")
            lines.extend(estimate_report(CACHE_PATH, estimates, max_bandwidth))

        for l in lines:
            print(l)

    def requirements_tree(
        self,
        override_params: Mapping[NamePattern, ParamValue]
        | Mapping[str, ParamValue]
        | None = None,
    ) -> ReqTree:
        """Compute the requirements tree for the given RUME.

        Parameters
        ----------
        override_params : Mapping[NamePattern, ParamValue], optional
            when computing requirements, use these values to override
            any that are provided by the RUME itself.  If keys are provided as strings,
            they must be able to be parsed as `NamePattern`s.

        Returns
        -------
        ReqTree
            the requirements tree
        """
        label_name, label_def = GEO_LABELS
        return ReqTree.of(
            requirements={
                # Start with our top-level requirements.
                **self.requirements,
                # Artificially require the geo labels attribute.
                label_name: label_def,
            },
            params=self._params_database(override_params),
        )

    def evaluate_params(
        self,
        rng: np.random.Generator,
        override_params: Mapping[NamePattern, ParamValue]
        | Mapping[str, ParamValue]
        | None = None,
    ) -> DataResolver:
        """
        Evaluates the parameters of this RUME.

        Parameters
        ----------
        rng : np.random.Generator, optional
            The random number generator to use during evaluation
        override_params : Mapping[NamePattern, ParamValue] | Mapping[str, ParamValue], optional
            Use these values to override any that are provided by the RUME itself.
            If keys are provided as strings, they must be able to be parsed as
            `NamePattern`s.

        Returns
        -------
        DataResolver
            the resolver containing the evaluated values
        """  # noqa: E501
        ps = None
        if override_params is not None and len(override_params) > 0:
            ps = {NamePattern.of(k): v for k, v in override_params.items()}

        reqs = self.requirements_tree(ps)
        return reqs.evaluate(self.dim, self.scope, rng)

    def _strata_dim(self, gpm: Gpm) -> SimDimensions:
        C = gpm.ipm.num_compartments
        E = gpm.ipm.num_events
        return dataclasses.replace(self.dim, compartments=C, events=E)

    def initialize(self, data: DataResolver, rng: np.random.Generator) -> SimArray:
        """
        Evaluates the Initializer(s) for this RUME.

        Parameters
        ----------
        data : DataResolver
            The resolved parameters for this RUME.
        rng : np.random.Generator
            The random number generator to use. Generally this should be the same
            RNG used to evaluate parameters.

        Returns
        -------
        SimArray
            the initial values (a NxC array) for all geo scope nodes and
            IPM compartments

        Raises
        ------
        InitException
            If initialization fails for any reason or produces invalid values.
        """
        try:
            return np.column_stack(
                [
                    gpm.init.with_context_internal(
                        namespace=ModuleNamespace(gpm_strata(gpm.name), "init"),
                        data=data,
                        dim=self._strata_dim(gpm),
                        scope=self.scope,
                        rng=rng,
                    ).evaluate()
                    for gpm in self.strata
                ]
            )
        except InitException as e:
            raise e
        except Exception as e:
            raise InitException("Initializer failed during evaluation.") from e


@dataclass(frozen=True)
class SingleStrataRume(Rume[GeoScopeT_co]):
    """A RUME with a single strata."""

    ipm: CompartmentModel

    @staticmethod
    def build(
        ipm: CompartmentModel,
        mm: MovementModel,
        init: Initializer,
        scope: GeoScopeT,
        time_frame: TimeFrame,
        params: Mapping[str, ParamValue],
    ) -> "SingleStrataRume[GeoScopeT]":
        """Create a RUME with only a single strata."""
        return SingleStrataRume(
            strata=[Gpm(DEFAULT_STRATA, ipm, mm, init)],
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
class MultistrataRume(Rume[GeoScopeT_co]):
    """A RUME with a multiple strata."""

    ipm: CombinedCompartmentModel

    @staticmethod
    def build(
        strata: Sequence[Gpm],
        meta_requirements: Sequence[AttributeDef],
        meta_edges: MetaEdgeBuilder,
        scope: GeoScopeT,
        time_frame: TimeFrame,
        params: Mapping[str, ParamValue],
    ) -> "MultistrataRume[GeoScopeT]":
        """Create a multistrata RUME by combining one GPM per strata."""
        return MultistrataRume(
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
    def meta_edges(self, symbols: MultistrataModelSymbols) -> Sequence[TransitionDef]:
        """
        When implementing a MultistrataRumeBuilder, override this method
        to build the meta-transition-edges -- the edges which represent
        cross-strata interactions. You are given a reference to this model's symbols
        library so you can build expressions for the transition rates.
        """

    @final
    def build(
        self,
        scope: GeoScopeT,
        time_frame: TimeFrame,
        params: Mapping[str, ParamValue],
    ) -> MultistrataRume[GeoScopeT]:
        """Build the RUME."""
        return MultistrataRume[GeoScopeT].build(
            self.strata,
            self.meta_requirements,
            self.meta_edges,
            scope,
            time_frame,
            params,
        )
