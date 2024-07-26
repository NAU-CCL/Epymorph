"""
A RUME (Runnable Modeling Experiment) is a package containing the critical components of an epymorph experiment.
Particular simulation tasks may require more information, but will certainly not require less.
A GPM (Geo-Population Model) is a subset of this configuration, and it is possible to combine multiple GPMs
into one multi-strata RUME.
"""
import dataclasses
import textwrap
from dataclasses import dataclass
from functools import cached_property
from itertools import accumulate
from typing import Callable, Mapping, OrderedDict, Self, Sequence

import numpy as np
from numpy.typing import NDArray
from sympy import Add, Expr, Max, Symbol

from epymorph.compartment_model import (CompartmentDef, CompartmentModel,
                                        ModelSymbols, TransitionDef,
                                        remap_transition)
from epymorph.data_shape import SimDimensions
from epymorph.data_type import dtype_str
from epymorph.database import AbsoluteName, ModuleNamePattern, NamePattern
from epymorph.geography.scope import GeoScope
from epymorph.initializer import Initializer
from epymorph.movement.parser import (DailyClause, MovementClause,
                                      MovementSpec, MoveSteps)
from epymorph.params import ParamSymbol, ParamValue, simulation_symbols
from epymorph.simulation import AttributeDef, TimeFrame
from epymorph.sympy_shim import to_symbol
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

    ipm: CompartmentModel
    mm: MovementSpec
    init: Initializer
    params: Mapping[ModuleNamePattern, ParamValue]

    def __init__(
        self,
        ipm: CompartmentModel,
        mm: MovementSpec,
        init: Initializer,
        params: Mapping[str, ParamValue] | None = None,
    ):
        self.ipm = ipm
        self.mm = mm
        self.init = init
        self.params = {
            ModuleNamePattern.parse(k): v
            for k, v in (params or {}).items()
        }


#####################################
# Utilities for building meta edges #
#####################################


class RumeSymbols:
    """
    A symbol dictionary for the symbols in a RUME. This information is made available during
    the meta-edge builder function so that you can reference the RUME symbols to create the
    appropriate transition rates.
    """
    _compartments: dict[str, tuple[Symbol, ...]]
    _attr: dict[str, tuple[Symbol, ...]]
    _meta: tuple[Symbol, ...]

    def __init__(
        self,
        compartments: dict[str, tuple[Symbol, ...]],
        attr: dict[str, tuple[Symbol, ...]],
        meta: tuple[Symbol, ...],
    ):
        self._compartments = compartments
        self._attr = attr
        self._meta = meta

    def compartments(self, strata: str) -> tuple[Symbol, ...]:
        """A tuple of symbols for the compartments in a strata."""
        return self._compartments[strata]

    def total(self, strata: str) -> Expr:
        """A sympy expression for the total of all compartments in a strata."""
        return Add(*self._compartments[strata])

    def total_nonzero(self, strata: str) -> Expr:
        """
        A sympy expression for the total of all compartments in a strata,
        but clamped so that it's never less than one. (This is useful as
        a divisor, if you can guarantee the numerator is zero when the
        sum otherwise would be zero.)
        """
        return Max(1, self.total(strata))

    def attributes(self, strata: str) -> tuple[Symbol, ...]:
        """A tuple of symbols for the (non-meta) IPM attributes in a strata."""
        return self._attr[strata]

    def meta_attributes(self) -> tuple[Symbol, ...]:
        """A tuple of symbols for the meta attributes."""
        return self._meta


MetaEdgeBuilder = Callable[[RumeSymbols], Sequence[TransitionDef]]
"""A function for creating meta edges in a multistrata RUME."""


########
# RUME #
########


DEFAULT_STRATA = "all"
"""The strata name used as the default, primarily for single-strata simulations."""
META_STRATA = "meta"
"""A strata for meta-strata information."""

GEO_LABELS = AbsoluteName(META_STRATA, "geo", "label")
"""
If this attribute is provided to a RUME, it will be used as labels for the geo node.
Otherwise we'll use the node IDs from the geo scope.
"""


@dataclass(frozen=True)
class _StrataSymbols(ModelSymbols):
    """The remapping of an IPM's symbols to use in a multi-strata IPM."""

    mapping: dict[Symbol, Symbol]

    @classmethod
    def map_to_strata(cls, symbols: ModelSymbols, strata: str) -> Self:
        """Remap an IPM's ModelSymbols to be in the given strata."""
        compartments = list[CompartmentDef]()
        attributes = OrderedDict[AbsoluteName, AttributeDef]()
        compartment_symbols = list[Symbol]()
        attribute_symbols = list[Symbol]()
        mapping = dict[Symbol, Symbol]()

        for comp, old_symbol in zip(symbols.compartments, symbols.compartment_symbols):
            new_name = f"{comp.name}_{strata}"
            new_symbol = to_symbol(new_name)
            mapping[old_symbol] = new_symbol
            compartments.append(dataclasses.replace(comp, name=new_name))
            compartment_symbols.append(new_symbol)

        for (name, attr), old_symbol in zip(symbols.attributes.items(), symbols.attribute_symbols):
            new_name = name.in_strata(f"gpm:{strata}")
            new_symbol = to_symbol(f"{attr.name}_{strata}")
            mapping[old_symbol] = new_symbol
            attributes[new_name] = attr
            attribute_symbols.append(new_symbol)

        return cls(compartments, attributes, compartment_symbols, attribute_symbols, mapping)


def combine_ipms(
    strata: list[tuple[str, CompartmentModel]],
    meta_attributes: list[AttributeDef],
    meta_edges: MetaEdgeBuilder,
) -> CompartmentModel:
    """
    Combine IPMs for different strata, remapping symbols as appropriate and using the
    `meta_edges` function to construct edges connecting the strata compartments.
    """

    strata_symbols = [
        _StrataSymbols.map_to_strata(ipm.symbols, strata)
        for strata, ipm in strata
    ]

    meta_attributes_symbols = [
        to_symbol(f"{a.name}_{META_STRATA}")
        for a in meta_attributes
    ]

    rume_symbols = RumeSymbols(
        compartments={
            strata: tuple(symbols.compartment_symbols)
            for (strata, _), symbols in zip(strata, strata_symbols)
        },
        attr={
            strata: tuple(symbols.attribute_symbols)
            for (strata, _), symbols in zip(strata, strata_symbols)
        },
        meta=tuple(meta_attributes_symbols),
    )

    ipm_symbols = ModelSymbols(
        compartments=[
            compartment
            for symbols in strata_symbols
            for compartment in symbols.compartments
        ],
        attributes=OrderedDict([
            *((name, attr)
              for symbols in strata_symbols
              for name, attr in symbols.attributes.items()),
            *((AbsoluteName(META_STRATA, "ipm", attr.name), attr)
              for attr in meta_attributes),
        ]),
        compartment_symbols=[
            compartment
            for symbols in strata_symbols
            for compartment in symbols.compartment_symbols
        ],
        attribute_symbols=[
            *(a for s in strata_symbols for a in s.attribute_symbols),
            *meta_attributes_symbols,
        ],
    )

    ipm_transitions = [
        *(remap_transition(trx, symbols.mapping)
            for (_, ipm), symbols in zip(strata, strata_symbols)
            for trx in ipm.transitions),
        *meta_edges(rume_symbols),
    ]

    return CompartmentModel(ipm_symbols, ipm_transitions)


def combine_tau_steps(strata_tau_lengths: dict[str, list[float]]) -> tuple[list[float], dict[str, dict[int, int]], dict[str, dict[int, int]]]:
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
    def tau_starts(taus: list[float]) -> list[float]:
        return [0.0, *accumulate(taus)][:-1]

    def tau_stops(taus: list[float]) -> list[float]:
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
    combined_tau_lengths = [
        stop - start
        for start, stop in zip(combined_tau_starts, combined_tau_stops)
    ]

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

    return combined_tau_lengths, tau_start_mapping, tau_stop_mapping


def remap_taus(strata_mms: list[tuple[str, MovementSpec]]) -> tuple[list[float], OrderedDict[str, MovementSpec]]:
    """
    When combining movement models with different tau steps, it is necessary to create a
    new tau step scheme which can accomodate them all.
    """
    new_tau_steps, start_mapping, stop_mapping = combine_tau_steps({
        strata: mm.steps.step_lengths
        for strata, mm in strata_mms
    })

    def clause_remap_tau(clause: MovementClause, strata: str) -> MovementClause:
        match clause:
            case DailyClause():
                return DailyClause(
                    days=clause.days,
                    leave_step=start_mapping[strata][clause.leave_step],
                    duration=clause.duration,
                    return_step=stop_mapping[strata][clause.return_step],
                    function=clause.function,
                )

    def spec_remap_tau(orig_spec: MovementSpec, strata: str) -> MovementSpec:
        return MovementSpec(
            steps=MoveSteps(new_tau_steps),
            attributes=orig_spec.attributes,
            predef=orig_spec.predef,
            clauses=[
                clause_remap_tau(c, strata)
                for c in orig_spec.clauses
            ],
        )

    return new_tau_steps, OrderedDict([
        (strata_name, spec_remap_tau(spec, strata_name))
        for strata_name, spec in strata_mms
    ])


class Rume:
    """
    A RUME (or Runnable Modeling Experiment) contains the configuration of an
    epymorph-style simulation. It brings together one or more IPMs, MMs, initialization routines,
    and a geo-temporal scope. Model parameters can also be specified on a RUME.
    The RUME will eventually be used to construct a Simulation, which is an
    algorithm that uses a RUME to produce some results -- in the most basic case,
    running a disease simulation and providing time-series results of the disease model.
    """

    original_gpms: OrderedDict[str, Gpm]
    ipm: CompartmentModel
    mms: OrderedDict[str, MovementSpec]
    scope: GeoScope
    time_frame: TimeFrame
    params: Mapping[NamePattern, ParamValue]
    dim: SimDimensions
    is_single_strata: bool

    def __init__(
        self,
        strata: OrderedDict[str, Gpm],
        ipm: CompartmentModel,
        mms: OrderedDict[str, MovementSpec],
        tau_step_lengths: list[float],
        scope: GeoScope,
        time_frame: TimeFrame,
        params: Mapping[NamePattern, ParamValue],
        is_single_strata: bool,
    ):
        """
        This is the 'internal' constructor for Rume; you probably want to use the
        `single_strata` or `multistrata` static methods instead.
        """
        if not are_unique(strata):
            msg = "Strata names must be unique; duplicate found."
            raise ValueError(msg)

        # Create dimensions
        dim = SimDimensions.build(
            tau_step_lengths=tau_step_lengths,
            start_date=time_frame.start_date,
            days=time_frame.duration_days,
            nodes=len(scope.get_node_ids()),
            compartments=ipm.num_compartments,
            events=ipm.num_events,
        )

        self.original_gpms = strata
        self.ipm = ipm
        self.mms = mms
        self.scope = scope
        self.time_frame = time_frame
        self.params = params
        self.dim = dim
        self.is_single_strata = is_single_strata

    @cached_property
    def attributes(self) -> Mapping[AbsoluteName, AttributeDef]:
        """Returns the attributes required by the RUME."""
        def generate_items():
            # IPM attributes are already fully named.
            yield from self.ipm.requirements_dict.items()
            # Name the MM and Init attributes.
            for strata, gpm in self.original_gpms.items():
                strata_name = f"gpm:{strata}"
                for a in gpm.mm.attributes:
                    yield AbsoluteName(strata_name, "mm", a.name), a
                for a in gpm.init.requirements:
                    yield AbsoluteName(strata_name, "init", a.name), a
        return dict(generate_items())

    def compartment_mask(self, strata_name: str) -> NDArray[np.bool_]:
        """
        Returns a mask which describes which compartments belong in the given strata.
        For example: if the model has three strata ('1', '2', and '3') with three compartments each,
        `strata_compartment_mask('2')` returns `[0 0 0 1 1 1 0 0 0]`
        (where 0 stands for False and 1 stands for True).
        Raises ValueError if no strata matches the given name.
        """
        result = np.full(shape=self.ipm.num_compartments,
                         fill_value=False, dtype=np.bool_)
        ci, cf = 0, 0
        found = False
        for strata, gpm in self.original_gpms.items():
            # Iterate through the strata IPMs:
            ipm = gpm.ipm
            if strata_name != strata:
                # keep count of how many compartments precede our target strata
                ci += ipm.num_compartments
            else:
                # when we find our target, we now have the target's compartment index range
                cf = ci + ipm.num_compartments
                # set those to True and break
                result[ci:cf] = True
                found = True
                break
        if not found:
            raise ValueError(f"Not a valid strata name in this model: {strata_name}")
        return result

    def compartment_mobility(self, strata_name: str) -> NDArray[np.bool_]:
        """Calculates which compartments should be considered subject to movement in a particular strata."""
        compartment_mobility = np.array(
            ['immobile' not in c.tags for c in self.ipm.compartments],
            dtype=np.bool_
        )
        return self.compartment_mask(strata_name) * compartment_mobility

    def with_time_frame(self, time_frame: TimeFrame) -> 'Rume':
        """Create a RUME with a new time frame."""
        # TODO: do we need to go through all of the params and subset any that are time-based?
        # How would that work? Or maybe reconciling to time frame happens at param evaluation time...
        return Rume(
            strata=self.original_gpms,
            ipm=self.ipm,
            mms=self.mms,
            tau_step_lengths=list(self.dim.tau_step_lengths),
            scope=self.scope,
            time_frame=time_frame,
            params=self.params,
            is_single_strata=self.is_single_strata,
        )

    def name_display_formatter(self) -> Callable[[AbsoluteName | NamePattern], str]:
        """Returns a function for formatting attribute/parameter names."""
        if self.is_single_strata:
            return lambda n: f"{n.module}::{n.id}"
        else:
            return str

    def params_description(self) -> str:
        """Provide a description of all attributes required by the RUME."""
        format_name = self.name_display_formatter()
        lines = []
        for name, attr in self.attributes.items():
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
        for name, attr in self.attributes.items():
            value = 'PLACEHOLDER'
            if attr.default_value is not None:
                value = str(attr.default_value)
            lines.append(f'    "{format_name(name)}": {value},')
        lines.append("}")
        return "\n".join(lines)

    @staticmethod
    def symbols(*symbols: ParamSymbol) -> tuple[Symbol, ...]:
        """Convenient function to retrieve the symbols used to represent simulation quantities."""
        return simulation_symbols(*symbols)

    @classmethod
    def single_strata(
        cls,
        ipm: CompartmentModel,
        mm: MovementSpec,
        init: Initializer,
        scope: GeoScope,
        time_frame: TimeFrame,
        params: Mapping[str, ParamValue],
    ) -> Self:
        """Create a RUME with only a single strata."""
        gpm = Gpm(ipm, mm, init, {})

        return cls(
            strata=OrderedDict([(DEFAULT_STRATA, gpm)]),
            ipm=ipm,
            mms=OrderedDict([(DEFAULT_STRATA, mm)]),
            tau_step_lengths=mm.steps.step_lengths,
            scope=scope,
            time_frame=time_frame,
            params={
                NamePattern.parse(k): v
                for k, v in params.items()
            },
            is_single_strata=True,
        )

    @classmethod
    def multistrata(
        cls,
        strata: list[tuple[str, Gpm]],
        meta_attributes: list[AttributeDef],
        meta_edges: MetaEdgeBuilder,
        scope: GeoScope,
        time_frame: TimeFrame,
        params: Mapping[str, ParamValue],
    ) -> Self:
        """Create a multi-strata RUME by combining GPMs, one for each strata."""
        # Combine IPMs
        ipm = combine_ipms(
            [(strata_name, gpm.ipm) for strata_name, gpm in strata],
            meta_attributes, meta_edges)

        # Combine MMs
        tau_step_lengths, mms = remap_taus(
            [(strata_name, gpm.mm) for strata_name, gpm in strata])

        return cls(
            strata=OrderedDict(strata),
            ipm=ipm,
            mms=mms,
            tau_step_lengths=tau_step_lengths,
            scope=scope,
            time_frame=time_frame,
            params={
                NamePattern.parse(k): v
                for k, v in params.items()
            },
            is_single_strata=False,
        )
