"""
Tools for rendering geographic maps from epymorph simulation output data.
"""

from functools import lru_cache
from itertools import repeat
from typing import Any, Callable, Iterable, cast

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import EngFormatter
from pyproj import CRS

from epymorph.compartment_model import QuantityAggregation, QuantitySelection
from epymorph.error import GeographyError
from epymorph.geography.scope import GeoAggregation, GeoGroup, GeoScope, GeoSelection
from epymorph.geography.us_census import CensusScope
from epymorph.geography.us_geography import STATE
from epymorph.geography.us_tiger import (
    get_block_groups_geo,
    get_counties_geo,
    get_states_geo,
    get_tracts_geo,
    is_tiger_year,
)
from epymorph.time import TimeAggregation, TimeSelection
from epymorph.tools.data import Output, munge


@lru_cache(16)
def _get_geo(scope: GeoScope) -> gpd.GeoDataFrame:
    if isinstance(scope, CensusScope) and is_tiger_year(scope.year):
        match scope.granularity:
            case "state":
                gdf = get_states_geo(scope.year)
            case "county":
                gdf = get_counties_geo(scope.year)
            case "tract":
                states = list({STATE.extract(x) for x in scope.node_ids})
                gdf = get_tracts_geo(scope.year, states)
            case "block group":
                states = list({STATE.extract(x) for x in scope.node_ids})
                gdf = get_block_groups_geo(scope.year, states)
            case _:
                raise GeographyError("Unsupported Census granularity.")
        return gpd.GeoDataFrame(gdf[gdf["GEOID"].isin(scope.node_ids)])

    raise GeographyError("Cannot draw choropleth maps for the given scope.")


class NodeLabelRenderer:
    """A class for rendering text labels on choropleth maps.
    The default implementation is very simple, but you may
    override this class to customize."""

    _color: str

    def __init__(self, color: str = "white"):
        self._color = color

    def coords(self, data_gdf: pd.DataFrame) -> gpd.GeoSeries:
        """Determine where to draw the labels."""
        # If we already have centroids in the gdf, use those.
        # Otherwise calculate from the polygon.
        # Note that this will throw a warning if a geographic CRS is used.
        return cast(
            gpd.GeoSeries,
            (
                data_gdf["centroid"]
                if "centroid" in data_gdf.columns
                else data_gdf.geometry.centroid
            ),
        )

    def labels(self, data_gdf: pd.DataFrame) -> Iterable[str | None]:
        """Determine the text of each label."""
        return data_gdf["data"].apply(lambda x: f"{x:.1f}")

    def colors(
        self,
        data_gdf: pd.DataFrame,
        color_scale: ScalarMappable,
    ) -> Iterable[str]:
        """Determine the color of each label."""
        return repeat(self._color, data_gdf["data"].size)

    def additional_kwargs(self, data_gdf: pd.DataFrame) -> Iterable[dict]:
        """Determine extra keyword arguments to pass to the `annotate()` method
        for each label."""
        return repeat(
            {"ha": "center", "va": "center", "fontsize": 8},
            data_gdf["data"].size,
        )

    def render(
        self,
        ax: Axes,
        data_gdf: pd.DataFrame,
        color_scale: ScalarMappable,
    ) -> None:
        """Render labels onto the given `matplotlib.axes.Axes`."""
        for point, label, color, kwargs in zip(
            self.coords(data_gdf),
            self.labels(data_gdf),
            self.colors(data_gdf, color_scale),
            self.additional_kwargs(data_gdf),
        ):
            if label is not None:
                ax.annotate(label, xy=(point.x, point.y), color=color, **kwargs)


class MapRenderer:
    """Provides a number of methods for rendering an output in choropleth map form."""

    output: Output

    def __init__(self, output: Output):
        self.output = output

    def geography_data(
        self,
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantitySelection | QuantityAggregation,
        *,
        proj: CRS | str | None = None,
        transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    ) -> gpd.GeoDataFrame:
        """Calculate the GeoDataFrame used for drawing maps
        merged with the output data from the given axis strategies."""

        # Our result should have three columns: time, node, and a data value
        data_df = munge(self.output, geo, time, quantity)
        if len(data_df["time"].unique()) > 1:
            err = (
                "When drawing a choropleth map, please ensure that you choose a "
                "time aggregation strategy that reduces the time series to a "
                "single point (scalar)."
            )
            raise ValueError(err)
        if len(data_df.columns) > 3:
            err = (
                "When drawing a choropleth map, please ensure that you choose a "
                "quantity strategy that produces a single value column -- either "
                "by selecting a single IPM quantity or aggregating multiple quantities "
                "to one value per geo node."
            )
            raise ValueError(err)
        data_df = data_df.set_axis(["time", "geo", "data"], axis=1)

        if transform is not None:
            data_df = transform(data_df)

        return gpd.GeoDataFrame(
            self.geography(geo, proj=proj).merge(
                left_on="GEOID",
                right_on="geo",
                right=data_df.iloc[:, [1, 2]],
            )
        )

    def geography(
        self,
        geo: GeoSelection | GeoGroup | GeoAggregation,
        *,
        proj: CRS | str | None = None,
    ) -> gpd.GeoDataFrame:
        """Calculate the GeoDataFrame for the given geo axis strategy, without merging
        any simulation data."""
        gdf = _get_geo(geo.to_scope())

        # If we have internal point info in the gdf, make it a centroid column.
        cols = gdf.columns
        if "INTPTLON" in cols and "INTPTLAT" in cols:
            centroids = gpd.points_from_xy(
                gdf["INTPTLON"],
                gdf["INTPTLAT"],
                crs=gdf.crs,
            )
            if proj is not None:
                centroids = centroids.to_crs(proj)
            gdf["centroid"] = centroids

        return cast(gpd.GeoDataFrame, gdf if proj is None else gdf.to_crs(proj))

    def choropleth(
        self,
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantitySelection | QuantityAggregation,
        *,
        borders: GeoSelection | GeoGroup | None = None,
        cmap: Any | None = None,
        proj: CRS | str | None = None,
        text_label: bool | str | NodeLabelRenderer = False,
        title: str | None = None,
        transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
        vmax: float | None = None,
        vmin: float | None = None,
    ) -> None:
        """
        Renders a choropleth map using GeoPandas and matplotlib showing the given
        selections. Selections must be made carefully to produce a valid map: the geo
        selection and grouping will dictate which polygons are shown on the map, the
        time selection must collapse to a single time-point, and the quantity selection
        must collapse to a single value per node. Of course there are many ways
        to trim the output data to that form, but we will check the result of the
        selections to verify it during rendering, and invalid selections will result
        in an raised exception.

        The plot will be immediately rendered by this function by calling `plt.show()`.
        This is intended as a quick plotting method to cover most casual use-cases.
        If you want more control over how the plot is drawn, see method
        `choropleth_plt()`.

        Parameters
        ----------
        geo : GeoSelection | GeoAggregation
            the geographic selection to make on the output data
        time : TimeSelection | TimeAggregation
            the time selection to make on the output data
        quantity : QuantitySelection | QuantityAggregation
            the quantity selection to make on the output data
        borders : GeoSelection | GeoGroup, optional
            if given, use this geography to draw dark borders,
            this could be the same or different geography from `geo`;
            if None (default), no borders are drawn
        cmap
            the color map to use for the plot; you can pass any value
            you would normally pass to `geopandas.GeoDataFrame.plot()`
        proj : CRS | str, optional
            the projection to use for mapping; if None (default) we will
            map using the default projection for the source geography
            (for US Census, this is NAD83 https://epsg.org/crs_4269/index.html)
        text_label : bool | str | NodeLabelRenderer, default=False
            True to render a text label of the data value for each polygon in white;
            a string to specify a different color, or a NodeLabelRenderer instance
            to use that.
        title : str, optional
            a title to draw on the plot
        transform : Callable[[pd.DataFrame], pd.DataFrame], optional
            allows you to specify an arbitrary transform function on the source
            dataframe before we plot it, e.g., to rescale the values.
            The dataframe given as the argument is the result of applying
            all selections and the projection if specified.
            You should return a dataframe with the same format, where the
            data column has been modified for your purposes.

            Dataframe columns:
            - "geo": the node ID of each polygon
            - "data": the data value from the quantity selection
        vmax : float, optional
            the max value for the color map, by default the max value of the data
        vmin : float, optional
            the min value for the color map, by default the min value of the data
        """
        try:
            fig, ax = plt.subplots(layout="constrained")
            ax.axis("off")

            data_gdf, color_scale = self.choropleth_plt(
                ax,
                geo,
                time,
                quantity,
                borders=borders,
                cmap=cmap,
                proj=proj,
                transform=transform,
                vmax=vmax,
                vmin=vmin,
            )

            # Draw colorbar
            fig.colorbar(mappable=color_scale, ax=ax, format=EngFormatter(sep=""))

            # Draw text labels on each polygon?
            if text_label is not False:
                if isinstance(text_label, str):
                    labeler = NodeLabelRenderer(text_label)
                elif isinstance(text_label, NodeLabelRenderer):
                    labeler = text_label
                else:
                    labeler = NodeLabelRenderer()
                labeler.render(ax, data_gdf, color_scale)

            if title is not None:
                plt.title(title)

            plt.show()
        except:
            plt.close()
            raise

    def choropleth_plt(
        self,
        ax: Axes,
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantitySelection | QuantityAggregation,
        *,
        borders: GeoSelection | GeoGroup | None = None,
        cmap: Any | None = None,
        proj: CRS | str | None = None,
        transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
        vmax: float | None = None,
        vmin: float | None = None,
    ) -> tuple[gpd.GeoDataFrame, ScalarMappable]:
        """
        Draws a choropleth map onto the given matplotlib Axes showing the given
        selections. This is a variant of the method `choropleth()` that gives you
        more control over the rendering of a plot by letting you do most of the work
        with matplotlib's API. See that method for conditions that must be met to
        use this method effectively.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            the plot axes on which to draw the map
        geo : GeoSelection | GeoAggregation
            the geographic selection to make on the output data
        time : TimeSelection | TimeAggregation
            the time selection to make on the output data
        quantity : QuantitySelection | QuantityAggregation
            the quantity selection to make on the output data
        borders : GeoSelection | GeoGroup, optional
            if given, use this geography to draw dark borders,
            this could be the same or different geography from `geo`;
            if None (default), no borders are drawn
        cmap
            the color map to use for the plot; you can pass any value
            you would normally pass to `geopandas.GeoDataFrame.plot()`
        proj : CRS | str, optional
            the projection to use for mapping; if None (default) we will
            map using the default projection for the source geography
            (for US Census, this is NAD83 https://epsg.org/crs_4269/index.html)
        transform : Callable[[pd.DataFrame], pd.DataFrame], optional
            allows you to specify an arbitrary transform function on the source
            dataframe before we plot it, e.g., to rescale the values.
            The dataframe given as the argument is the result of applying
            all selections and the projection if specified.
            You should return a dataframe with the same format, where the
            data column has been modified for your purposes.

            Dataframe columns:
            - "geo": the node ID of each polygon
            - "data": the data value from the quantity selection
        vmax : float, optional
            the max value for the color map, by default the max value of the data
        vmin : float, optional
            the min value for the color map, by default the min value of the data

        Returns
        -------
        tuple[GeoDataFrame, ScalarMappable]
            a tuple with 1. the GeoDataFrame containing the data used to render the map
            and 2. the ScalarMappable used as the map's color scale
        """
        data_gdf = self.geography_data(
            geo,
            time,
            quantity,
            proj=proj,
            transform=transform,
        )

        vmin = vmin if vmin is not None else cast(float, data_gdf["data"].min())
        vmax = vmax if vmax is not None else cast(float, data_gdf["data"].max())
        color_scale = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=Normalize(vmin, vmax),
        )

        # Draw polygons.
        data_gdf.plot(
            ax=ax,
            column="data",
            legend=False,
            cmap=color_scale.cmap,
            vmin=color_scale.norm.vmin,
            vmax=color_scale.norm.vmax,
        )

        # Draw borders?
        if borders is not None:
            borders_gdf = self.geography(borders, proj=proj)
            borders_gdf.plot(
                ax=ax,
                linewidth=1,
                edgecolor="black",
                color="none",
                alpha=0.8,
                legend=False,
            )

        return data_gdf, color_scale


class MapRendererMixin(Output):
    """Mixin class that adds a convenient method for rendering choropleth maps
    from an output."""

    @property
    def map(self) -> MapRenderer:
        """Render a choropleth map from this output."""
        return MapRenderer(self)
