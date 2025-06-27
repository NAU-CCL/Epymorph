"""
A RUME (Runnable Modeling Experiment) is a package containing the critical components
of an epymorph experiment. Particular simulation tasks may require more information,
but will certainly not require less. A GPM (Geo-Population Model) is a subset of this
configuration, and it is possible to combine multiple GPMs into one multi-strata RUME.
"""

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
    Sequence,
    TypeVar,
    final,
)

import numpy as np
from numpy.typing import NDArray
from sympy import Symbol
from typing_extensions import override

from epymorph.attribute import (
    AbsoluteName,
    AttributeDef,
    ModuleNamePattern,
    NamePattern,
)
from epymorph.cache import CACHE_PATH
from epymorph.compartment_model import (
    BaseCompartmentModel,
    CombinedCompartmentModel,
    CompartmentModel,
    MetaEdgeBuilder,
    MultiStrataModelSymbols,
    TransitionDef,
)
from epymorph.data_shape import Shapes
from epymorph.data_type import SimArray, dtype_str
from epymorph.data_usage import CanEstimateData, estimate_report
from epymorph.database import (
    Database,
    DatabaseWithFallback,
    DatabaseWithStrataFallback,
    DataResolver,
    ReqTree,
)
from epymorph.error import InitError
from epymorph.geography.scope import GeoScope
from epymorph.initializer import Initializer
from epymorph.movement_model import MovementClause, MovementModel
from epymorph.params import ParamSymbol, simulation_symbols
from epymorph.simulation import (
    Context,
    ParamValue,
    SimulationFunction,
    TickDelta,
    TickIndex,
)
from epymorph.strata import DEFAULT_STRATA, META_STRATA, gpm_strata
from epymorph.time import TimeFrame
from epymorph.util import (
    CovariantMapping,
    KeyValue,
    are_unique,
    map_values,
)

#######
# GPM #
#######


@dataclass(frozen=True)
class GPM:
    """
    A GPM (short for Geo-Population Model) combines an IPM, MM, and
    initialization scheme. Most often, a GPM is used to define one strata
    in a multi-strata RUME.

    Parameters
    ----------
    name :
        The name to use to identify the GPM.
    ipm :
        The IPM for the GPM.
    mm :
        The MM for the GPM.
    init :
        The initializer for the GPM.
    params :
        Parameter values specific to this GPM. When a GPM is used in a RUME,
        RUME parameters will override GPM parameters if there's overlap.

    See Also
    --------
    [epymorph.rume.MultiStrataRUME.build][] which uses GPMs.
    """

    name: str
    """The name to use to identify the GPM."""
    ipm: CompartmentModel
    """The IPM for the GPM."""
    mm: MovementModel
    """The MM for the GPM."""
    init: Initializer
    """The initializer for the GPM."""
    params: Mapping[ModuleNamePattern, ParamValue] | None = field(default=None)
    """
    Parameter values specific to this GPM. When a GPM is used in a RUME, RUME parameters
    will override GPM parameters if there's overlap.
    """

    # NOTE: constructing a ModuleNamePattern object is a bit awkward from an interface
    # perspective; much more ergonomic to just be able to use strings -- but that
    # requires a parsing call. Doing that parsing here is awkward for a dataclass.
    # And we could design around that but I'm not certain this feature isn't destinated
    # to be removed anyway... so for now users will have to do the parsing or maybe
    # we'll add a utility function that effectively does this:
    # params = {ModuleNamePattern.parse(k): v for k, v in (params or {}).items()}  # noqa: E501, ERA001


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
A special attribute which, if provided to a RUME ("meta::geo::label"), will be used as
labels for the geo node. Otherwise labels will be taken from the geo scope.
"""


class CombineTauStepsResult(NamedTuple):
    """The result of the `combine_tau_steps` function."""

    new_tau_steps: tuple[float, ...]
    """The lengths of the output tau steps."""
    start_mapping: dict[str, dict[int, int]]
    """
    A per-strata mapping for the start of tau step bounds;
    from original tau step index to new tau step index.
    """
    stop_mapping: dict[str, dict[int, int]]
    """
    A per-strata mapping for the end of tau step bounds;
    from original tau step index to new tau step index.
    """


def combine_tau_steps(
    strata_tau_lengths: dict[str, Sequence[float]],
) -> CombineTauStepsResult:
    """
    Combine multiple tau step schemes to calculate a unified scheme which is compatible
    with all of them while using the fewest possible number of tau steps.

    Parameters
    ----------
    strata_tau_lengths :
        A mapping containing each strata name (key) and the list of tau lengths (value).

    Returns
    -------
    :
        A tuple containing the information required to adjust movement models to the
        combined tau step scheme.

    Examples
    --------
    For tau steps `[1/3, 2/3]` and tau steps `[1/2, 1/2]` --
    the combined tau steps are `[1/3, 1/6, 1/2]`.
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

    return CombineTauStepsResult(
        combined_tau_lengths, tau_start_mapping, tau_stop_mapping
    )


def remap_taus(
    strata_mms: list[tuple[str, MovementModel]],
) -> OrderedDict[str, MovementModel]:
    """
    Adjust a set of movement models which may use different tau step schemes such that
    they all use a unified tau step scheme which is compatible with all of them.

    This step is performed automatically when constructing a multi-strata RUME.

    Parameters
    ----------
    strata_mms :
        The list of tuples of strata name and the movement model for that strata.

    Returns
    -------
    :
        The adjusted movement model for each strata.
    """
    new_tau_steps, start_mapping, stop_mapping = combine_tau_steps(
        {strata: mm.steps for strata, mm in strata_mms}
    )

    def clause_remap_tau(clause: MovementClause, strata: str) -> MovementClause:
        leave_step = start_mapping[strata][clause.leaves.step]
        return_step = (
            stop_mapping[strata][clause.returns.step]
            if clause.returns.step >= 0
            else -1  # "never"
        )

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
"""A type of `GeoScope`."""
GeoScopeT_co = TypeVar("GeoScopeT_co", covariant=True, bound=GeoScope)
"""A type of `GeoScope`, covariant."""


@dataclass(frozen=True)
class RUME(ABC, Generic[GeoScopeT_co]):
    """
    A RUME (or Runnable Modeling Experiment) contains the configuration of an
    epymorph-style simulation. It brings together one or more IPMs, MMs, initialization
    routines, and a geo-temporal scope. Model parameters can also be specified.

    RUMEs are often used to construct a simulation -- in the most basic case, running a
    forward simulation and producing time-series results of disease progression.

    `RUME` is generic on the type of `GeoScope` used to construct it (`GeoScopeT_co`).

    See Also
    --------
    `RUME` is an abstract parent class; users will typically use
    [epymorph.rume.SingleStrataRUME.build][] and
    [epymorph.rume.MultiStrataRUMEBuilder][] to construct concrete RUMEs.
    """

    strata: Sequence[GPM]
    """The strata for the RUME expressed as GPMs."""
    ipm: BaseCompartmentModel
    """The effective IPM for the RUME, made by combining all strata IPMs."""
    mms: OrderedDict[str, MovementModel]
    """ The effective MMs for the RUME by strata, made by combining all strata MMs."""
    scope: GeoScopeT_co
    """The geo scope. This is shared by all strata."""
    time_frame: TimeFrame
    """The simulation time frame."""
    params: Mapping[NamePattern, ParamValue]
    """Parameter values for the RUME."""

    tau_step_lengths: list[float] = field(init=False)
    """The lengths of each tau step in the simulation as fractions of a day."""
    num_tau_steps: int = field(init=False)
    """The number of tau steps per day in the simulation."""
    num_ticks: int = field(init=False)
    """
    The number of total simulation ticks, the product of multiplying
    the number of simulation days from the time frame by the number of tau steps
    per day.
    """

    def __post_init__(self):
        if not are_unique(g.name for g in self.strata):
            msg = "Strata names must be unique; duplicate found."
            raise ValueError(msg)

        # We can get the tau step lengths from a movement model.
        # In a multistrata model, there will be multiple remapped MMs,
        # but they all have the same set of tau steps so it doesn't matter
        # which we use. (Using the first one is safe.)
        first_strata = self.strata[0].name
        steps = self.mms[first_strata].steps
        object.__setattr__(self, "tau_step_lengths", steps)
        object.__setattr__(self, "num_tau_steps", len(steps))
        object.__setattr__(self, "num_ticks", len(steps) * self.time_frame.days)

    @cached_property
    def requirements(self) -> Mapping[AbsoluteName, AttributeDef]:
        """The attributes required by the RUME."""

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
        override_params: CovariantMapping[str | NamePattern, ParamValue] | None = None,
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
        `strata_compartment_mask('b')` returns `[F F F T T T F F F]`.
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

        Currently a compartment is exempted from movement calculations if it is tagged
        "immobile".
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
        """
        Create function for formatting attribute/parameter names.

        Returns
        -------
        :
            The formatter function.
        """

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
        Retrieve the sympy symbols used to represent simulation quantities.

        Parameters
        ----------
        *symbols :
            The symbols to retrieve, as var-args.

        Returns
        -------
        :
            A tuple containing the symbols requested, in the order
            requested.
        """
        return simulation_symbols(*symbols)

    def estimate_data(
        self,
        *,
        max_bandwidth: int = 1000**2,  # default: 1 MB/s
    ) -> None:
        """
        Print a report estimating the data requirements of this RUME.

        Includes data which must be downloaded and how much will be added to the file
        cache. Provides a projected download time based on the given assumed maximum
        network bandwidth.

        Parameters
        ----------
        max_bandwidth :
            The assumed maximum network download speed in bytes per second.
            Default is 1 MB/s.
        """

        ctx = Context.of(
            scope=self.scope,
            time_frame=self.time_frame,
            ipm=self.ipm,
        )
        estimates = [
            p.with_context_internal(ctx).estimate_data()
            for p in self.params.values()
            if isinstance(p, SimulationFunction) and isinstance(p, CanEstimateData)
        ]

        lines = list[str]()
        if len(estimates) == 0:
            lines.append("ADRIO data usage is either negligible or non-estimable.")
        else:
            lines.append("ADRIO data usage estimation:")
            lines.extend(estimate_report(CACHE_PATH, estimates, max_bandwidth))

        for l in lines:
            print(l)  # noqa: T201

    def requirements_tree(
        self,
        override_params: CovariantMapping[str | NamePattern, ParamValue] | None = None,
    ) -> ReqTree:
        """
        Compute the requirements tree for the given RUME.

        Parameters
        ----------
        override_params :
            When computing requirements, use these values to override
            any that are provided by the RUME itself. If keys are provided as strings,
            they must be able to be parsed as `NamePattern`s.

        Returns
        -------
        :
            The requirements tree.

        Raises
        ------
        DataAttributeError
            If the tree cannot be evaluated, for instance, due to containing circular
            dependencies.
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
        override_params: CovariantMapping[str | NamePattern, ParamValue] | None = None,
    ) -> DataResolver:
        """
        Evaluate the parameters of this RUME.

        Parameters
        ----------
        rng :
            The random number generator to use during evaluation
        override_params :
            Use these values to override any that are provided by the RUME itself.
            If keys are provided as strings, they must be able to be parsed as
            `NamePattern`s.

        Returns
        -------
        :
            The resolver containing the evaluated values.

        Raises
        ------
        DataAttributeError
            If the parameters cannot be evaluated for any reason, such as missing or
            invalid parameter values.
        """
        ps = None
        if override_params is not None and len(override_params) > 0:
            ps = {NamePattern.of(k): v for k, v in override_params.items()}

        reqs = self.requirements_tree(ps)
        return reqs.evaluate(self.scope, self.time_frame, self.ipm, rng)

    def initialize(self, data: DataResolver, rng: np.random.Generator) -> SimArray:
        """
        Evaluate the Initializer(s) for this RUME.

        Parameters
        ----------
        data :
            The resolved parameters for this RUME.
        rng :
            The random number generator to use. Generally this should be the same
            RNG used to evaluate parameters.

        Returns
        -------
        :
            the initial values ((N,C)-shaped array) for all geo scope nodes and
            IPM compartments

        Raises
        ------
        InitError
            If initialization fails for any reason or produces invalid values.
        """

        def gpm_ctx(gpm: GPM) -> Context:
            return Context.of(
                name=AbsoluteName(gpm_strata(gpm.name), "init", "init"),
                data=data,
                scope=self.scope,
                time_frame=self.time_frame,
                ipm=gpm.ipm,
                rng=rng,
            )

        try:
            return np.column_stack(
                [
                    gpm.init.with_context_internal(gpm_ctx(gpm)).evaluate()
                    for gpm in self.strata
                ]
            )
        except InitError as e:
            raise e
        except Exception as e:
            raise InitError("Initializer failed during evaluation.") from e


@dataclass(frozen=True)
class SingleStrataRUME(RUME[GeoScopeT_co]):
    """
    A RUME with a single strata. We recommend using the more-convenient static method
    `SingleStrataRUME.build` instead of the normal class constructor.

    `SingleStrataRUME` is generic on the type of `GeoScope` used to construct it
    (`GeoScopeT_co`).
    """

    ipm: CompartmentModel

    @staticmethod
    def build(
        ipm: CompartmentModel,
        mm: MovementModel,
        init: Initializer,
        scope: GeoScopeT,
        time_frame: TimeFrame,
        params: CovariantMapping[str | NamePattern, ParamValue],
    ) -> "SingleStrataRUME[GeoScopeT]":
        """
        Create a RUME with a single strata.

        This method is generic on the type of `GeoScope` used (`GeoScopeT`).

        Parameters
        ----------
        ipm :
            The compartmental model.
        mm :
            The movement model.
        init :
            The logic for setting the initial conditions of the simulation.
        scope :
            The geo scope.
        time_frame :
            The time frame to simulate.
        params :
            Parameter values that will be used to fulfill the data requirements
            of the various modules of the RUME.

        Returns
        -------
        :
            The RUME instance.
        """
        return SingleStrataRUME(
            strata=[GPM(DEFAULT_STRATA, ipm, mm, init)],
            ipm=ipm,
            mms=OrderedDict([(DEFAULT_STRATA, mm)]),
            scope=scope,
            time_frame=time_frame,
            params={NamePattern.of(k): v for k, v in params.items()},
        )

    @override
    def name_display_formatter(self) -> Callable[[AbsoluteName | NamePattern], str]:
        return lambda n: f"{n.module}::{n.id}"


@dataclass(frozen=True)
class MultiStrataRUME(RUME[GeoScopeT_co]):
    """
    A RUME with multiple strata.

    `MultiStrataRUME` is generic on the type of `GeoScope` used to construct it
    (`GeoScopeT_co`).

    See Also
    --------
    [epymorph.rume.MultiStrataRUMEBuilder][] for a more user-friendly way to construct
    multistrata RUMEs.
    """

    ipm: CombinedCompartmentModel

    @staticmethod
    def build(
        strata: Sequence[GPM],
        meta_requirements: Sequence[AttributeDef],
        meta_edges: MetaEdgeBuilder,
        scope: GeoScopeT,
        time_frame: TimeFrame,
        params: CovariantMapping[str | NamePattern, ParamValue],
    ) -> "MultiStrataRUME[GeoScopeT]":
        """
        Create a multistrata RUME by combining one GPM per strata.

        This function is not the recommended way to create multistrata RUMEs;
        see `MultiStrataRUMEBuilder` instead.

        This method is generic on the type of `GeoScope` used (`GeoScopeT`).

        Parameters
        ----------
        strata :
            Define the strata for this RUME by providing a GPM for each.
        meta_requirements :
            Define any data requirements used for the meta-edges of the combined IPM.
        meta_edges :
            A function which constructs the meta-edges of the combined IPM.
        scope :
            The geo scope.
        time_frame :
            The time frame to simulate.
        params :
            Parameter values that will be used to fulfill the data requirements
            of the various modules of the RUME.

        Returns
        -------
        :
            The RUME instance.
        """
        return MultiStrataRUME(
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
            params={NamePattern.of(k): v for k, v in params.items()},
        )

    @override
    def name_display_formatter(self) -> Callable[[AbsoluteName | NamePattern], str]:
        return str


class MultiStrataRUMEBuilder(ABC):
    """
    The recommended way to define and create multistrata RUMEs. Implement this class
    and then call `build` to obtain an instance.
    """

    strata: Sequence[GPM]
    """The strata that are part of this RUME."""

    meta_requirements: Sequence[AttributeDef]
    """
    A set of additional requirements which are needed by the meta-edges
    in our combined compartment model.
    """

    @abstractmethod
    def meta_edges(self, symbols: MultiStrataModelSymbols) -> Sequence[TransitionDef]:
        """
        When implementing a MultiStrataRumeBuilder, override this method
        to define the meta-transition-edges -- the edges which represent
        cross-strata interactions.

        Parameters
        ----------
        symbols :
            The model's symbols library, to obtain compartment and parameter symbols
            needed to build transition rate expressions.

        Returns
        -------
        :
            The set of transitions to add to the resulting combined IPM.
        """

    @final
    def build(
        self,
        scope: GeoScopeT,
        time_frame: TimeFrame,
        params: CovariantMapping[str | NamePattern, ParamValue],
    ) -> MultiStrataRUME[GeoScopeT]:
        """
        Complete the RUME definition and construct an instance.

        This method is generic on the type of `GeoScope` used (`GeoScopeT`).

        Parameters
        ----------
        scope :
            The geo scope.
        time_frame :
            The time frame to simulate.
        params :
            Parameter values that will be used to fulfill the data requirements
            of the various modules of the RUME.

        Returns
        -------
        :
            The RUME instance.
        """
        return MultiStrataRUME[GeoScopeT].build(
            self.strata,
            self.meta_requirements,
            self.meta_edges,
            scope,
            time_frame,
            params,
        )
