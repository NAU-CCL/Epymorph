"""ADRIOs that access the US Census TIGER geography files."""

from typing import cast

import numpy as np
from numpy.typing import NDArray
from pandas import to_numeric
from typing_extensions import override

from epymorph.adrio.adrio import (
    ADRIO,
    ADRIOContextError,
    InspectResult,
    adrio_cache,
    adrio_validate_pipe,
)
from epymorph.adrio.validation import (
    ResultFormat,
    on_structured,
    validate_dtype,
    validate_numpy,
    validate_shape,
    validate_values_in_range,
)
from epymorph.data_shape import Shapes
from epymorph.data_type import CentroidDType, StructDType
from epymorph.data_usage import AvailableDataEstimate
from epymorph.error import MissingContextError
from epymorph.geography.us_census import CensusScope, StateScope
from epymorph.geography.us_tiger import check_cache, is_tiger_year
from epymorph.simulation import Context


class _USTIGERMixin(ADRIO):
    """Common ADRIO logic for US TIGER ADRIOs."""

    @override
    def validate_context(self, context: Context) -> None:
        try:
            scope = context.scope  # scope is required
        except MissingContextError as e:
            raise ADRIOContextError(self, context, str(e))
        if not isinstance(scope, CensusScope):
            err = "Census scope is required for us_tiger attributes."
            raise ADRIOContextError(self, context, err)
        year = scope.year
        if not is_tiger_year(year):
            err = f"{year} is not a supported year for us_tiger attributes."
            raise ADRIOContextError(self, context, err)

    @override
    def estimate_data(self) -> AvailableDataEstimate:
        scope = cast(CensusScope, self.scope)
        est = check_cache(scope.granularity, scope.year)
        return AvailableDataEstimate(
            name=self.class_name,
            cache_key=f"us_tiger:{scope.granularity}:{scope.year}",
            new_network_bytes=est.missing_cache_size,
            new_cache_bytes=est.missing_cache_size,
            total_cache_bytes=est.total_cache_size,
            max_bandwidth=None,
        )


@adrio_cache
class GeometricCentroid(_USTIGERMixin, ADRIO[StructDType, np.float64]):
    """The centroid of the geographic polygons."""

    # FROM MIXIN:
    # def validate_context(self, context: Context) -> None:
    # def estimate_data(self) -> AvailableDataEstimate:

    @property
    @override
    def result_format(self) -> ResultFormat:
        return ResultFormat(shape=Shapes.N, dtype=CentroidDType)

    @override
    def validate_result(self, context: Context, result: NDArray) -> None:
        adrio_validate_pipe(
            self,
            context,
            result,
            validate_numpy(),
            validate_shape(self.result_format.shape.to_tuple(context.dim)),
            validate_dtype(self.result_format.dtype),
            on_structured(validate_values_in_range(-180.0, +180.0)),
        )

    @override
    def inspect(self) -> InspectResult[StructDType, np.float64]:
        self.validate_context(self.context)
        scope = cast(CensusScope, self.scope)

        source_gdf = scope.geography
        result_np = (
            source_gdf["geometry"]
            .apply(lambda x: x.centroid.coords[0])
            .to_numpy(dtype=CentroidDType)
        )

        self.validate_result(self.context, result_np)
        return InspectResult(
            self,
            source=source_gdf,
            result=result_np,
            dtype=self.result_format.dtype.type,
            shape=self.result_format.shape,
            issues={},
        )


@adrio_cache
class InternalPoint(_USTIGERMixin, ADRIO[StructDType, np.float64]):
    """
    The internal point provided by TIGER data. These points are selected by
    Census workers so as to be guaranteed to be within the geographic polygons,
    while geometric centroids have no such guarantee.
    """

    # FROM MIXIN:
    # def validate_context(self, context: Context) -> None:
    # def estimate_data(self) -> AvailableDataEstimate:

    @property
    @override
    def result_format(self) -> ResultFormat:
        return ResultFormat(shape=Shapes.N, dtype=CentroidDType)

    @override
    def validate_result(self, context: Context, result: NDArray) -> None:
        adrio_validate_pipe(
            self,
            context,
            result,
            validate_numpy(),
            validate_shape(self.result_format.shape.to_tuple(context.dim)),
            validate_dtype(self.result_format.dtype),
            on_structured(validate_values_in_range(-180.0, +180.0)),
        )

    @override
    def inspect(self) -> InspectResult[StructDType, np.float64]:
        self.validate_context(self.context)
        scope = cast(CensusScope, self.scope)

        info_df = scope.get_info()
        centroids = list(
            zip(
                to_numeric(info_df["INTPTLON"]),
                to_numeric(info_df["INTPTLAT"]),
            )
        )
        result_np = np.array(centroids, dtype=CentroidDType)
        self.validate_result(self.context, result_np)

        return InspectResult(
            self,
            source=info_df,
            result=result_np,
            dtype=self.result_format.dtype.type,
            shape=self.result_format.shape,
            issues={},
        )


@adrio_cache
class Name(_USTIGERMixin, ADRIO[np.str_, np.str_]):
    """For states and counties, the proper name of the location; otherwise its GEOID."""

    # FROM ADRIO:
    # def validate_result(self, context: Context, result: NDArray) -> None:
    # FROM MIXIN:
    # def validate_context(self, context: Context) -> None:
    # def estimate_data(self) -> AvailableDataEstimate:

    @property
    @override
    def result_format(self) -> ResultFormat:
        return ResultFormat(shape=Shapes.N, dtype=np.str_)

    @override
    def inspect(self) -> InspectResult[np.str_, np.str_]:
        self.validate_context(self.context)
        scope = cast(CensusScope, self.scope)

        # NOTE: info isn't really needed for Tract/CBG scope, but if we special case
        # to avoid loading it then we have to duplicate this logic in cache estimation
        # and invent a "source" for the inspect result. I think better to just load it.
        info_df = scope.get_info()

        if scope.granularity in ("state", "county"):
            result_np = info_df["NAME"].to_numpy(dtype=np.str_)
        else:
            # There aren't good names for Tracts or CBGs, just use GEOID
            result_np = scope.node_ids

        self.validate_result(self.context, result_np)
        return InspectResult(
            self,
            source=info_df,
            result=result_np,
            dtype=self.result_format.dtype.type,
            shape=self.result_format.shape,
            issues={},
        )


@adrio_cache
class PostalCode(_USTIGERMixin, ADRIO[np.str_, np.str_]):
    """
    For states only, the postal code abbreviation for the state
    ("AZ" for Arizona, and so on).
    """

    # FROM ADRIO:
    # def validate_result(self, context: Context, result: NDArray) -> None:
    # FROM MIXIN:
    # def estimate_data(self) -> AvailableDataEstimate:

    @property
    @override
    def result_format(self) -> ResultFormat:
        return ResultFormat(shape=Shapes.N, dtype=np.str_)

    @override
    def validate_context(self, context: Context) -> None:
        if not isinstance(context.scope, StateScope):
            err = "PostalCode is only available for StateScopes."
            raise ADRIOContextError(self, context, err)
        year = context.scope.year
        if not is_tiger_year(year):
            err = f"{year} is not a supported year for us_tiger attributes."
            raise ADRIOContextError(self, context, err)

    @override
    def inspect(self) -> InspectResult[np.str_, np.str_]:
        self.validate_context(self.context)
        scope = cast(CensusScope, self.scope)

        info_df = scope.get_info()
        result_np = info_df["STUSPS"].to_numpy(dtype=np.str_)

        self.validate_result(self.context, result_np)
        return InspectResult(
            self,
            source=info_df,
            result=result_np,
            dtype=self.result_format.dtype.type,
            shape=self.result_format.shape,
            issues={},
        )


@adrio_cache
class LandAreaM2(_USTIGERMixin, ADRIO[np.float64, np.float64]):
    """
    The land area of the geo node in meters-squared. This is the 'ALAND' attribute
    from the TIGER data files.
    """

    # FROM ADRIO:
    # def validate_result(self, context: Context, result: NDArray) -> None:
    # FROM MIXIN:
    # def validate_context(self, context: Context) -> None:
    # def estimate_data(self) -> AvailableDataEstimate:

    @property
    @override
    def result_format(self) -> ResultFormat:
        return ResultFormat(shape=Shapes.N, dtype=np.float64)

    @override
    def inspect(self) -> InspectResult[np.float64, np.float64]:
        self.validate_context(self.context)
        scope = cast(CensusScope, self.scope)

        info_df = scope.get_info()
        result_np = info_df["ALAND"].to_numpy(dtype=np.float64)

        self.validate_result(self.context, result_np)
        return InspectResult(
            self,
            source=info_df,
            result=result_np,
            dtype=self.result_format.dtype.type,
            shape=self.result_format.shape,
            issues={},
        )
