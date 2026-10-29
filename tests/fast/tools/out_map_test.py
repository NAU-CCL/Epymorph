import geopandas as gpd
import matplotlib.pyplot as plt
import pytest
from shapely.geometry import box

# See conftest for RUME and Output fixtures


def test_choropleth_plt(rume, output, monkeypatch):
    geography = gpd.GeoDataFrame(
        {"GEOID": ["35", "04"]},
        geometry=[box(1, 0, 2, 1), box(0, 0, 1, 1)],
        crs="EPSG:4326",
    )
    renderer = output.map
    monkeypatch.setattr(renderer, "geography", lambda geo, proj=None: geography)
    fig, ax = plt.subplots()
    try:
        data, color_scale = renderer.choropleth_plt(
            ax,
            rume.scope.select.all(),
            rume.time_frame.select.days(1, 1),
            rume.ipm.select.compartments("S"),
            transform=lambda df: df.assign(data=df["data"] / 100),
        )
        assert data["geo"].tolist() == ["35", "04"]
        assert data["data"].tolist() == [33, 32]
        assert color_scale.norm.vmin == 32
        assert color_scale.norm.vmax == 33
        assert len(ax.collections) == 1
    finally:
        plt.close(fig)


def test_choropleth_rejects_multiple_times_or_quantities(rume, output):
    geo = rume.scope.select.all()
    with pytest.raises(ValueError, match="time aggregation"):
        output.map.geography_data(
            geo,
            rume.time_frame.select.all(),
            rume.ipm.select.compartments("S"),
        )
    with pytest.raises(ValueError, match="single value column"):
        output.map.geography_data(
            geo,
            rume.time_frame.select.days(1, 1),
            rume.ipm.select.compartments(),
        )
