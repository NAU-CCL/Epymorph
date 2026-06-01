"""
Simulators for performing multi-realization simulations such as forecasting and
state/parameter fitting.
"""

import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Mapping, Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from epymorph.attribute import NamePattern
from epymorph.compartment_model import (
    CompartmentDef,
    QuantityStrategy,
    TransitionDef,
)
from epymorph.data_type import SimDType
from epymorph.forecasting.dynamic_params import ParamFunctionDynamics, Prior
from epymorph.forecasting.munge_realizations import (
    ParameterSelector,
    ParameterStrategy,
    RealizationAggregation,
    RealizationSelection,
    RealizationSelector,
)
from epymorph.geography.scope import GeoAggregation, GeoSelection
from epymorph.rume import RUME
from epymorph.simulator.basic.output import Output
from epymorph.time import (
    Dim,
    TimeAggregation,
    TimeFrame,
    TimeSelection,
)
from epymorph.tools.data import mask


@dataclass(frozen=True)
class UnknownParam:
    """
    Contains the information for an unknown parameter. An unknown parameter is a
    parameter which can vary across realizations in a multi-realization simulation. Some
    simulators will try to estimate unknown parameters.

    Parameters
    ----------
    prior :
        The prior distribution or initial values of the parameter.
    dynamics :
        The dynamics of the parameter dictating how the parameter changes over time.
    """

    prior: NDArray[np.float64] | Prior
    """
    The prior distribution or initial values of the parameter.
    """

    dynamics: ParamFunctionDynamics
    """
    The dynamics of the parameter dictating how the parameter changes over time.
    """


@dataclass(frozen=True)
class PipelineConfig:
    """
    Contains the basic information needed to initialize a `PipelineSimulator`.
    Some simulators may require additional information provided in their respective
    constructor.

    Parameters
    ----------
    rume :
        The RUME used by the simulator.
    num_realizations :
        The number of realizations of the simulator.
    initial_values :
        The optional array of initial compartment values of the simulator. It has shape
        (R, N, C) where R is the number of realizations, N is the number of nodes,
        and C is the number of compartments.
    unknown_params :
        The dictionary of unknown paramters of the simulator.
    """

    rume: RUME
    """
    The RUME used by the simulator.
    """

    num_realizations: int
    """
    The number of realizations of the simulator.
    """

    initial_values: NDArray[SimDType] | None
    """
    The optional array of initial compartment values of the simulator.
    """

    unknown_params: Mapping[NamePattern, UnknownParam]
    """
    The dictionary of unknown paramters of the simulator.
    """

    @classmethod
    def from_rume(
        cls,
        rume: RUME,
        num_realizations: int,
        initial_values: NDArray[SimDType] | None = None,
        unknown_params: Mapping[str | NamePattern, UnknownParam]
        | Mapping[str, UnknownParam] = {},
    ) -> Self:
        """
        Creates a PipelineConfig from a RUME. Converts the keys of unknown_params into
        NamePattern's.

        Parameters
        ----------
        rume :
            The RUME used by the simulator.
        num_realizations :
            The number of realizations of the simulator.
        initial_values :
            The optional array of initial compartment values of the simulator. It has
            shape (R, N, C) where R is the number of realizations, N is the number of
            nodes, and C is the number of compartments.
        unknown_params :
            The dictionary of unknown paramters of the simulator. String names are
            converted to NamePattern
        """
        return cls(
            rume=rume,
            num_realizations=num_realizations,
            initial_values=initial_values,
            unknown_params={NamePattern.of(k): v for k, v in unknown_params.items()},
        )

    @classmethod
    def from_output(
        cls,
        output: "PipelineOutput",
        extend_duration: int,
        override_dynamics: Mapping[str | NamePattern, ParamFunctionDynamics]
        | Mapping[str, ParamFunctionDynamics] = {},
    ) -> Self:
        """
        Creates a PipelineConfig starting from the end of a previous PipelineOutput.
        The RUME, unknown parameters, and initial compartment values are taken from the
        output. The time frame is extended from the end of the previous time frame.

        Parameters
        ----------
        output :
            The output to extend.
        extend_duration :
            The number of days to extend the simulation from the end of the previous
            output.
        override_dynamics :
            Override the dyanmics of any parameters from the previous output.
        """
        new_time_frame = TimeFrame.of(
            start_date=output.rume.time_frame.end_date + datetime.timedelta(1),
            duration_days=extend_duration,
        )
        rume = replace(output.rume, time_frame=new_time_frame)
        unknown_params = {
            k: UnknownParam(prior=output.final_params[k], dynamics=v.dynamics)
            for k, v in output.unknown_params.items()
        }
        override_dynamics = {NamePattern.of(k): v for k, v in override_dynamics.items()}
        for k, v in override_dynamics.items():
            name = NamePattern.of(k)
            unknown_params[name] = UnknownParam(
                prior=unknown_params[name].prior, dynamics=v
            )
        return cls(
            rume=rume,
            num_realizations=output.num_realizations,
            initial_values=output.final_compartments,
            unknown_params=unknown_params,
        )


@dataclass(frozen=True)
class PipelineOutput:
    """
    The output of a `PipelineSimulator` run.
    """

    simulator: "PipelineSimulator"
    """
    The simulator used to generate the output. It contains information like the RUME,
    the (prior) unknown parameters,  the number of realizations, and how the
    compartments were initialized.
    """

    final_compartments: NDArray[SimDType]
    """
    An array of shape (R, N, C) where R is the number of realizations, N is the number
    of nodes, and C is the number of compartments. Each realization is an array of
    compartment values suitable for intializing another simulation. The precise
    interpretation of the array depends on the simulator used to produce the output.
    """

    final_params: Mapping[NamePattern, NDArray[np.float64]]
    """
    A dictionary where the keys are unknown parameters and the values are of arrays of
    shape (R, N) where R is the number of realizations and N is the number of nodes.
    Each realization of a parameter is an array of parameter values suitable for
    intializing another simulation. The precise interpretation of the array depends on
    the simulator used to produce the output.
    """

    compartments: NDArray[SimDType]
    """
    An array of shape (R, S, N, C) where R is the number of realizations, S is
    the number of tau steps, N is the number of nodes, and C is the number of
    compartments. The interpretation of this output depends on the simulator
    which produced it. For example, each realization could represent a full trajectory
    of the model. Alternatively it could represent a sampled estimate of the compartment
    values with no guarantees of the temporal structure.
    """

    events: NDArray[SimDType]
    """
    An array of shape (R, S, N, E) where R is the number of realizations, S is
    the number of tau steps, N is the number of nodes, and E is the number of
    events. The interpretation of this output depends on the simulator
    which produced it. For example, each realization could represent a full trajectory
    of the model. Alternatively it could represent a sampled estimate of the compartment
    values with no guarantees of the temporal structure.
    """

    initial: NDArray[SimDType]
    """
    An array of shape (R, N, C) where R is the number of realizations, N is the
    number of nodes, and C is the number of compartments. The interpretation of this
    output depends on the simulator which produced it.
    """

    estimated_params: Mapping[NamePattern, NDArray[np.float64]]
    """
    A dictionary where the keys are unknown parameters and the values are
    arrays of shape (R, T, N) where R is the number of realizations, T is the number of
    days and N is the number of nodes. Each realization of a parameter is an array of
    parameter values suitable for intializing another simulation. The precise
    interpretation of the array depends on the simulator used to produce the output.
    """

    @property
    def rume(self) -> RUME:
        return self.simulator.rume

    @property
    def unknown_params(self) -> Mapping[NamePattern, UnknownParam]:
        return self.simulator.unknown_params

    @property
    def num_realizations(self) -> int:
        return self.simulator.num_realizations

    @property
    def initial_values(self) -> NDArray[SimDType] | None:
        return self.simulator.initial_values

    @property
    def num_unknown_parameters(self) -> int:
        return len(self.unknown_params.keys())

    @property
    def dataframe(self) -> pd.DataFrame:
        NP = self.num_realizations  # noqa: N806
        C = self.rume.ipm.num_compartments
        E = self.rume.ipm.num_events
        N = self.rume.scope.nodes
        S = self.rume.num_ticks
        P = len(self.unknown_params.keys())  # noqa: N806
        tau_steps = self.rume.num_tau_steps

        states_np = np.concatenate(
            (self.compartments, self.events),
            axis=3,
        )

        ### Extract the parameter data
        param_np = np.concatenate(
            [
                np.repeat(val[..., np.newaxis], tau_steps, axis=1)
                for val in self.estimated_params.values()
            ],
            axis=3,
        )

        data_np = np.concatenate((states_np, param_np), axis=-1).reshape(
            (-1, P + E + C)
        )

        # Here I'm concatting two DFs sideways so that the index columns come first.
        # Could use insert, but this is nicer.
        return pd.concat(
            (
                # A dataframe for the various indices
                pd.DataFrame(
                    {
                        "realization": np.repeat(np.arange(NP), S * N),
                        "tick": np.tile(np.repeat(np.arange(S), N), NP),
                        "date": np.tile(
                            np.repeat(self.rume.time_frame.to_numpy(), N * tau_steps),
                            NP,
                        ),
                        "node": np.tile(self.rume.scope.node_ids, S * NP),
                    }
                ),
                # A dataframe for the data columns
                pd.DataFrame(
                    data=data_np,
                    columns=[
                        *(c.name.full for c in self.rume.ipm.compartments),
                        *(e.name.full for e in self.rume.ipm.events),
                        *(str(key) for key in self.unknown_params.keys()),
                    ],
                ),
            ),
            axis=1,  # stick them together side-by-side
        )

    @property
    def select(self) -> RealizationSelector:
        """
        Returns an instance of RealizationSelector.
        """
        return RealizationSelector(self.num_realizations)

    @property
    def param_select(self) -> ParameterSelector:
        return ParameterSelector(self.unknown_params)  # type: ignore


@dataclass(frozen=True)
class PipelineSimulator(ABC):
    """
    A base class for multi-realization simulations.
    """

    config: PipelineConfig

    @property
    def rume(self) -> RUME:
        """
        The RUME containing the IPM, MM, scope, time_frame, and the fixed params for
        each realization.
        """
        return self.config.rume

    @property
    def num_realizations(self) -> int:
        """
        The number of realzations.
        """
        return self.config.num_realizations

    @property
    def initial_values(self) -> NDArray[SimDType] | None:
        """
        An optional array of initial compartment values of shape (R, N, C) where R is
        the number of realizations, N is the number of nodes, and C is the number of
        compartments. If not specified, the initializer contained in the RUME will be
        used to generate a single initial condition which will be used for all
        realizations.
        """
        return self.config.initial_values

    @property
    def unknown_params(self) -> Mapping[NamePattern, UnknownParam]:
        """
        An array of unknown parameters which will vary across realizations.
        """
        return self.config.unknown_params

    @abstractmethod
    def run(self, rng: np.random.Generator) -> PipelineOutput:
        """
        Run the simulation.
        """


def munge_pipeline_output(
    output: PipelineOutput,
    realization: RealizationSelection | RealizationAggregation,
    geo: GeoSelection | GeoAggregation,
    time: TimeSelection | TimeAggregation,
    quantity: QuantityStrategy | ParameterStrategy,
) -> pd.DataFrame:
    NP = output.num_realizations  # noqa: N806
    N = output.rume.scope.nodes
    S = output.rume.num_ticks
    C = output.rume.ipm.num_compartments
    E = output.rume.ipm.num_events
    taus = output.rume.num_tau_steps
    P = output.num_unknown_parameters  # noqa: N806

    # Apply selections first so that aggregations operate on less data.
    time_mask = np.tile(np.repeat(mask(S, time.selection_ticks(taus)), N), NP)
    geo_mask = np.tile(np.tile(geo.selection, S), NP)

    # columns are: ["realization","tick", "date", "node", *quantities]

    columns = [True, True, True, True]
    if isinstance(quantity, QuantityStrategy):
        columns = np.concatenate(
            (columns, quantity.selection, [False for _ in range(P)])
        )
    else:
        columns = np.concatenate(
            (columns, [False for _ in range(C + E)], quantity.selection)
        )

    data_df = output.dataframe
    data_df = data_df.loc[time_mask & geo_mask].loc[:, columns]
    data_df = data_df.set_axis(
        ["realization", "tick", "date", "geo", *data_df.columns[4:]], axis=1
    )

    # This is a potential pain point, a multi-index would be far more
    # efficient, but would require reworking the selectors.
    data_df = data_df[data_df["realization"].isin(realization.selection)]

    if geo.aggregation is None:
        # Without agg: use node IDs as the geo dimension.
        pass
    else:
        # With agg:
        agg_df = data_df
        if geo.grouping is None:
            # With no grouping: the geo dimension collapses.
            geo_groups = "*"
        else:
            # With group: geo dimension comes from group.
            geo_groups = geo.grouping.map(agg_df["geo"].to_numpy())

        data_df = (
            data_df.assign(geo=geo_groups)
            .groupby(["realization", "tick", "date", "geo"], sort=False)
            .agg(geo.aggregation)
            .reset_index()
        )

    if quantity.aggregation is None:
        # For the sake of aggregating and sorting, we need to ensure column names
        # are not ambiguous. But we don't want to alter the names arbitrarily;
        # so we'll rename the columns, do our munging, then restore the original names.
        q_mapping = quantity.disambiguate()
        data_df = data_df.set_axis([*data_df.columns[0:4], *q_mapping.keys()], axis=1)
    else:
        # currently only supported agg is "sum"
        def agg(qty_indices: tuple[int, ...]) -> pd.Series:
            offset = 4  # we have three leading columns before quantities start
            col_indices = [i + offset for i in qty_indices]
            return data_df.iloc[:, col_indices].sum(axis=1)

        group_defs, group_indices = quantity.grouping.map(quantity.selected)  # type: ignore
        group_names = [g.name.full for g in group_defs]
        data_df = pd.DataFrame(
            {
                "realization": data_df["realization"],
                "tick": data_df["tick"],
                "date": data_df["date"],
                "geo": data_df["geo"],
                **{g: agg(xs) for g, xs in zip(group_names, group_indices)},
            }
        )
        # When there is any grouping applied, it should not be possible to
        # produce ambiguous quantity names; but to keep things simple we'll
        # just provide a no-op map for this case.
        q_mapping = dict(zip(group_names, group_names))

    if time.aggregation is None:
        # Without agg: drop date and use ticks as the time dimension.
        data_df = data_df.drop(columns=["date"]).rename(columns={"tick": "time"})
    else:
        # With agg:
        if time.grouping is None:
            # Without group: time dimension collapses.
            time_axis = "*"
        else:
            # With group: time dimension comes from grouping.
            nodes = data_df["geo"].unique().shape[0]
            sample_realizations = data_df["realization"].unique().shape[0]
            days = data_df.shape[0] // (sample_realizations * nodes * taus)
            time_axis = time.grouping.map(
                Dim(nodes=nodes, days=days, tau_steps=taus),
                data_df["tick"].to_numpy(),
                data_df["date"].to_numpy(),
            )

            if time_axis.shape[0] != data_df.shape[0]:
                err = "Chosen time-axis grouping did not return a group for every row."
                raise ValueError(err)

        # We need to perform different time aggregations based on the
        # quantity of interest. Currently the aggregation for UnknownParam
        # and TransitionDef is identical(but this may be subject to change).
        time_aggs = {}
        for col, q in zip(q_mapping.keys(), quantity.selected):
            if isinstance(q, CompartmentDef):
                time_aggs[col] = time.aggregation.compartments
            elif isinstance(q, UnknownParam):
                time_aggs[col] = time.aggregation.parameters
            elif isinstance(q, TransitionDef):
                time_aggs[col] = time.aggregation.events
            else:
                raise ValueError(f"Invalid quantity selector {str(q)}")

        data_df = (
            data_df.drop(columns=["tick", "date"])
            .assign(time=time_axis)
            .groupby(["realization", "time", "geo"], sort=False)
            .agg(time_aggs)
            .reset_index()
        )

    if realization.aggregation is None:
        # Without agg: use realization IDs as the realization dimension.
        pass
    else:
        # This avoids adding aggregation over the realization column to the dataframe.
        value_cols = list(q_mapping.keys())

        agg_dict = {f"{col_name}": realization.aggregation for col_name in value_cols}

        data_df = (
            data_df.groupby(["time", "geo"], sort=False)[value_cols]
            .agg(agg_dict)
            .reset_index()
        )

    return data_df.rename(columns=q_mapping).reset_index(drop=True)


def munge_combined(
    output: PipelineOutput | Output,
    geo: GeoSelection | GeoAggregation,
    time: TimeSelection | TimeAggregation,
    quantity: QuantityStrategy | ParameterStrategy,
    realization: RealizationSelection | RealizationAggregation | None = None,
) -> pd.DataFrame:
    data_df = output.dataframe
    NP = 1  # noqa: N806
    P = 0  # noqa: N806
    if isinstance(output, Output):
        data_df.insert(0, "realization", 0)
        realization = RealizationSelector(NP).all()
    else:
        NP = output.num_realizations  # noqa: N806
        P = output.num_unknown_parameters  # noqa: N806

    N = output.rume.scope.nodes
    S = output.rume.num_ticks
    C = output.rume.ipm.num_compartments
    E = output.rume.ipm.num_events
    taus = output.rume.num_tau_steps

    # Apply selections first so that aggregations operate on less data.
    time_mask = np.tile(np.repeat(mask(S, time.selection_ticks(taus)), N), NP)
    geo_mask = np.tile(np.tile(geo.selection, S), NP)

    # columns are: ["realization","tick", "date", "node", *quantities]
    columns = [True, True, True, True]
    if isinstance(quantity, QuantityStrategy):
        columns = np.concatenate(
            (columns, quantity.selection, [False for _ in range(P)])
        ).astype(bool)
    else:
        columns = np.concatenate(
            (columns, [False for _ in range(C + E)], quantity.selection)
        ).astype(bool)

    data_df = data_df.loc[time_mask & geo_mask].loc[:, columns]
    data_df = data_df.set_axis(
        ["realization", "tick", "date", "geo", *data_df.columns[4:]], axis=1
    )
    data_df = data_df.loc[data_df["realization"].isin(realization.selection)]  # type: ignore

    if geo.aggregation is None:
        # Without agg: use node IDs as the geo dimension.
        pass
    else:
        # With agg:
        agg_df = data_df
        if geo.grouping is None:
            # With no grouping: the geo dimension collapses.
            geo_groups = "*"
        else:
            # With group: geo dimension comes from group.
            geo_groups = geo.grouping.map(agg_df["geo"].to_numpy())

        data_df = (
            data_df.assign(geo=geo_groups)
            .groupby(["realization", "tick", "date", "geo"], sort=False)
            .agg(geo.aggregation)
            .reset_index()
        )

    if quantity.aggregation is None:
        # For the sake of aggregating and sorting, we need to ensure column names
        # are not ambiguous. But we don't want to alter the names arbitrarily;
        # so we'll rename the columns, do our munging, then restore the original names.
        q_mapping = quantity.disambiguate()
        data_df = data_df.set_axis([*data_df.columns[0:4], *q_mapping.keys()], axis=1)
    else:
        # currently only supported agg is "sum"
        def agg(qty_indices: tuple[int, ...]) -> pd.Series:
            offset = 4  # we have three leading columns before quantities start
            col_indices = [i + offset for i in qty_indices]
            return data_df.iloc[:, col_indices].sum(axis=1)

        group_defs, group_indices = quantity.grouping.map(quantity.selected)  # type: ignore
        group_names = [g.name.full for g in group_defs]
        data_df = pd.DataFrame(
            {
                "realization": data_df["realization"],
                "tick": data_df["tick"],
                "date": data_df["date"],
                "geo": data_df["geo"],
                **{g: agg(xs) for g, xs in zip(group_names, group_indices)},
            }
        )
        # When there is any grouping applied, it should not be possible to
        # produce ambiguous quantity names; but to keep things simple we'll
        # just provide a no-op map for this case.
        q_mapping = dict(zip(group_names, group_names))

    if time.aggregation is None:
        # Without agg: drop date and use ticks as the time dimension.
        data_df = data_df.drop(columns=["date"]).rename(columns={"tick": "time"})
    else:
        # With agg:
        if time.grouping is None:
            # Without group: time dimension collapses.
            time_axis = "*"
        else:
            # With group: time dimension comes from grouping.
            nodes = data_df["geo"].unique().shape[0]
            sample_realizations = data_df["realization"].unique().shape[0]
            days = data_df.shape[0] // (sample_realizations * nodes * taus)
            time_axis = time.grouping.map(
                Dim(nodes=nodes, days=days, tau_steps=taus),
                data_df["tick"].to_numpy(),
                data_df["date"].to_numpy(),
            )

            if time_axis.shape[0] != data_df.shape[0]:
                err = "Chosen time-axis grouping did not return a group for every row."
                raise ValueError(err)

        time_aggs = {}
        for col, q in zip(q_mapping.keys(), quantity.selected):
            if isinstance(q, CompartmentDef):
                time_aggs[col] = time.aggregation.compartments
            elif isinstance(q, UnknownParam):
                time_aggs[col] = time.aggregation.parameters
            else:
                time_aggs[col] = time.aggregation.events

        data_df = (
            data_df.drop(columns=["tick", "date"])
            .assign(time=time_axis)
            .groupby(["realization", "time", "geo"], sort=False)
            .agg(time_aggs)
        )

    if realization.aggregation is None:  # type: ignore
        # If there is only a single realization drop the realization column
        if NP == 1:
            data_df = data_df.drop(columns=["realization"])
    else:
        # This avoids adding aggregation over the realization column to the dataframe.
        value_cols = list(q_mapping.keys())

        agg_dict = {f"{col_name}": realization.aggregation for col_name in value_cols}  # type: ignore

        data_df = data_df.groupby(["time", "geo"], sort=False)[value_cols].agg(agg_dict)

    return data_df.rename(columns=q_mapping).reset_index(drop=True)
