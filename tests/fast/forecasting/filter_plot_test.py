import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgb
from matplotlib.container import BarContainer
from matplotlib.patches import Polygon
from scipy.stats import gaussian_kde

from epymorph.forecasting.filter_plot import PlotRendererPipeline


def test_pipeline_output_plot(rume, filter_output):
    renderer = filter_output.plot
    assert isinstance(renderer, PlotRendererPipeline)
    assert renderer.output is filter_output

    fig, ax = plt.subplots()
    try:
        lines = renderer.line_plt(
            ax,
            filter_output.select.all().agg(["mean"]),
            rume.scope.select.by_state("04"),
            rume.time_frame.select.all(),
            rume.ipm.select.compartments("S"),
        )
        assert len(lines) == 1
        assert lines[0].get_label() == "AZ: S: mean"
        np.testing.assert_array_equal(lines[0].get_xdata(), [1, 2, 3, 4])
        np.testing.assert_array_equal(lines[0].get_ydata(), [1, 1, 1, 1])
    finally:
        plt.close(fig)


##############
# line plots #
##############


@pytest.mark.parametrize(
    ("ordering", "expected_groups"),
    [
        ("location", ["AZ: S", "AZ: I", "NM: S", "NM: I"]),
        ("quantity", ["AZ: S", "NM: S", "AZ: I", "NM: I"]),
    ],
)
def test_line_plt_ordering_and_transform(
    rume, filter_output, ordering, expected_groups
):
    seen = []

    def transform(data):
        seen.append((list(data.columns), data["metric"].iloc[0]))
        return data.assign(value=data["value"] * 2)

    fig, ax = plt.subplots()
    try:
        lines = filter_output.plot.line_plt(
            ax,
            filter_output.select.all().agg(["std", "mean"]),
            rume.scope.select.all(),
            rume.time_frame.select.all(),
            rume.ipm.select.compartments("S", "I"),
            ordering=ordering,
            line_kwargs=[{"color": "red"}, {"color": "blue"}, {"color": "green"}],
            transform=transform,
        )
        assert [line.get_label() for line in lines] == [
            f"{group}: {metric}"
            for group in expected_groups
            for metric in ("mean", "std")
        ]
        assert [line.get_color() for line in lines] == [
            "red",
            "blue",
            "green",
            "red",
            "blue",
            "green",
            "red",
            "blue",
        ]
        assert seen == [
            (["metric", "value"], metric)
            for _ in expected_groups
            for metric in ("mean", "std")
        ]
        np.testing.assert_array_equal(lines[0].get_ydata(), [2, 2, 2, 2])
        np.testing.assert_array_equal(lines[1].get_ydata(), [0, 0, 0, 0])
        assert ax.get_legend() is None
        assert ax.get_xlabel() == ""
    finally:
        plt.close(fig)


@pytest.mark.parametrize("method", ["line", "line_plt"])
def test_line_rejects_realization_selection(rume, filter_output, method):
    fig, ax = plt.subplots()
    try:
        args = (
            filter_output.select.all(),
            rume.scope.select.all(),
            rume.time_frame.select.all(),
            rume.ipm.select.compartments("S"),
        )
        if method == "line_plt":
            args = (ax, *args)
        with pytest.raises(ValueError, match="only support RealizationAggregation"):
            getattr(filter_output.plot, method)(*args)
    finally:
        plt.close(fig)


def test_line_ordering_and_legend(rume, filter_output, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    try:
        filter_output.plot.line(
            filter_output.select.all().mean(),
            rume.scope.select.all(),
            rume.time_frame.select.all(),
            rume.ipm.select.compartments("S", "I"),
            ordering="quantity",
            label_format="{q} @ {n}: {m}",
            line_kwargs=[{"color": "red"}, {"color": "blue"}],
            legend="on",
        )
        ax = plt.gca()
        assert ax.get_legend_handles_labels()[1] == [
            "S @ AZ: mean",
            "S @ NM: mean",
            "I @ AZ: mean",
            "I @ NM: mean",
        ]
        assert [line.get_color() for line in ax.lines] == ["red", "blue", "red", "blue"]
        assert ax.get_legend() is not None
    finally:
        plt.close()


def test_line_plt_parameter_aggregation_and_date(rume, filter_output):
    fig, ax = plt.subplots()
    try:
        lines = filter_output.plot.line_plt(
            ax,
            filter_output.select.all().agg(["quantile_2.5", "mean"]),
            rume.scope.select.all().sum(),
            rume.time_frame.select.all(),
            filter_output.param_select.by_name("test_0"),
            label_format="{q} @ {n}: {m}",
            line_kwargs=[],
            time_format="date",
        )
        assert [line.get_label() for line in lines] == [
            "*::*::test_0 @ *: mean",
            "*::*::test_0 @ *: quantile_2.5",
        ]
        pd.testing.assert_index_equal(
            pd.DatetimeIndex(lines[0].get_xdata()),
            pd.date_range("2021-01-02", periods=4),
        )
        np.testing.assert_array_equal(lines[0].get_ydata(), [2, 2, 2, 2])
    finally:
        plt.close(fig)


def test_line_plt_disambiguates_duplicate_labels(rume, filter_output, monkeypatch):
    quantity = rume.ipm.select.compartments("S", "I")
    monkeypatch.setattr(
        type(quantity),
        "disambiguate_groups",
        lambda self: {"S_0": "same", "I_1": "same"},
    )
    fig, ax = plt.subplots()
    try:
        lines = filter_output.plot.line_plt(
            ax,
            filter_output.select.all().mean(),
            rume.scope.select.by_state("04"),
            rume.time_frame.select.all(),
            quantity,
        )
        assert [line.get_label() for line in lines] == [
            "AZ: same: mean",
            "AZ: same: mean",
        ]
    finally:
        plt.close(fig)


###################
# spaghetti plots #
###################


@pytest.mark.parametrize(
    ("ordering", "expected_groups"),
    [
        ("location", ["AZ: S", "AZ: I", "NM: S", "NM: I"]),
        ("quantity", ["AZ: S", "NM: S", "AZ: I", "NM: I"]),
    ],
)
def test_spaghetti_plt_ordering_and_transform(
    rume, filter_output, ordering, expected_groups
):
    fig, ax = plt.subplots()
    seen = []

    def transform(data):
        seen.append((data["geo"].iloc[0], data["quantity"].iloc[0]))
        return data.assign(time=data["time"] + 10, value=data["value"] * 2)

    try:
        lines = filter_output.plot.spaghetti_plt(
            ax,
            filter_output.select.all(),
            rume.scope.select.all(),
            rume.time_frame.select.all(),
            rume.ipm.select.compartments("S", "I"),
            ordering=ordering,
            line_kwargs=[{"color": "red"}, {"color": "blue"}],
            transform=transform,
        )

        assert len(lines) == 12
        assert list(ax.lines) == lines
        assert [line.get_label() for line in lines[::3]] == expected_groups
        assert all(
            line.get_label() == "_nolegend_"
            for index, line in enumerate(lines)
            if index % 3 != 0
        )
        assert [line.get_color() for line in lines] == [
            color for color in ("red", "blue", "red", "blue") for _ in range(3)
        ]
        assert [f"{geo}: {quantity}" for geo, quantity in seen] == [
            group for group in expected_groups for _ in range(3)
        ]
        np.testing.assert_array_equal(lines[0].get_xdata(), [11, 12, 13, 14])
        np.testing.assert_array_equal(lines[0].get_ydata(), [2, 2, 2, 2])
        assert ax.get_legend() is None
        assert ax.get_xlabel() == ""
    finally:
        plt.close(fig)


def test_spaghetti_plt_subplots_and_single_realization(rume, filter_output):
    fig, axes = plt.subplots(1, 2)
    realization = filter_output.select.sample(rng=np.random.default_rng(42), size=1)
    try:
        for ax, state, expected in zip(axes, ("04", "35"), ("AZ", "NM")):
            lines = filter_output.plot.spaghetti_plt(
                ax,
                realization,
                rume.scope.select.by_state(state),
                rume.time_frame.select.all(),
                rume.ipm.select.compartments("S"),
                label_format="{q} @ {n}",
                time_format="date",
            )
            assert len(lines) == 1
            assert lines[0].get_label() == f"S @ {expected}"
            assert ax.get_legend_handles_labels()[1] == [f"S @ {expected}"]
            pd.testing.assert_index_equal(
                pd.DatetimeIndex(lines[0].get_xdata()),
                pd.date_range("2021-01-02", periods=4),
            )
    finally:
        plt.close(fig)


@pytest.mark.parametrize("line_kwargs", [None, [], [{"color": None}], [{"c": None}]])
def test_spaghetti_plt_default_group_colors_and_parameters(
    rume, filter_output, line_kwargs
):
    fig, ax = plt.subplots()
    try:
        lines = filter_output.plot.spaghetti_plt(
            ax,
            filter_output.select.all(),
            rume.scope.select.all(),
            rume.time_frame.select.all(),
            filter_output.param_select.by_name("test_0"),
            line_kwargs=line_kwargs,
        )
        assert len(lines) == 6
        assert len({line.get_color() for line in lines[:3]}) == 1
        assert len({line.get_color() for line in lines[3:]}) == 1
        assert lines[0].get_color() != lines[3].get_color()
        assert ax.get_legend_handles_labels()[1] == [
            "AZ: *::*::test_0",
            "NM: *::*::test_0",
        ]
    finally:
        plt.close(fig)


@pytest.mark.parametrize("method", ["spaghetti", "spaghetti_plt"])
def test_spaghetti_rejects_realization_aggregation(rume, filter_output, method):
    fig, ax = plt.subplots()
    try:
        args = (
            filter_output.select.all().mean(),
            rume.scope.select.all(),
            rume.time_frame.select.all(),
            rume.ipm.select.compartments("S"),
        )
        if method == "spaghetti_plt":
            args = (ax, *args)
        with pytest.raises(ValueError, match="only support RealizationSelection"):
            getattr(filter_output.plot, method)(*args)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("legend", ["auto", "on", "off", "outside"])
def test_spaghetti_legend(rume, filter_output, monkeypatch, legend):
    monkeypatch.setattr(plt, "show", lambda: None)
    try:
        filter_output.plot.spaghetti(
            filter_output.select.all(),
            rume.scope.select.all(),
            rume.time_frame.select.all(),
            rume.ipm.select.compartments("S", "I"),
            label_format="{q} @ {n}",
            ordering="quantity",
            line_kwargs=[{"alpha": 0.3}],
            legend=legend,
            time_format="date",
            title="Forecast",
        )
        fig = plt.gcf()
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert ax.get_title() == "Forecast"
        assert ax.get_xlabel() == "date"
        assert ax.get_ylabel() == "count"
        assert ax.get_legend_handles_labels()[1] == [
            "S @ AZ",
            "S @ NM",
            "I @ AZ",
            "I @ NM",
        ]
        if legend == "off":
            assert ax.get_legend() is None
        else:
            plot_legend = ax.get_legend()
            assert plot_legend is not None
            if legend == "outside":
                assert plot_legend.get_bbox_to_anchor() is not None
    finally:
        plt.close()


def test_spaghetti_save_single_plot(rume, filter_output, tmp_path):
    path = tmp_path / "spaghetti.png"
    try:
        filter_output.plot.spaghetti(
            filter_output.select.all(),
            rume.scope.select.all(),
            rume.time_frame.select.all(),
            rume.ipm.select.compartments("S"),
            to_file=path,
        )
        assert path.is_file()
        assert path.stat().st_size > 0
    finally:
        plt.close()


@pytest.mark.parametrize(("groups", "has_legend"), [(12, True), (13, False)])
def test_spaghetti_auto_legend_counts_groups(
    rume, filter_output, monkeypatch, groups, has_legend
):
    renderer = filter_output.plot
    monkeypatch.setattr(plt, "show", lambda: None)

    def draw_groups(ax, *args, **kwargs):
        lines = []
        for group in range(groups):
            for realization in range(3):
                label = f"group {group}" if realization == 0 else "_nolegend_"
                lines.extend(ax.plot([0, 1], [0, 1], label=label))
        return lines

    monkeypatch.setattr(renderer, "spaghetti_plt", draw_groups)
    try:
        renderer.spaghetti(
            filter_output.select.all(),
            rume.scope.select.all(),
            rume.time_frame.select.all(),
            rume.ipm.select.compartments("S"),
        )
        assert (plt.gca().get_legend() is not None) is has_legend
    finally:
        plt.close()


def test_spaghetti_closes_figure_on_failure(rume, filter_output):
    args = (
        filter_output.select.all(),
        rume.scope.select.all(),
        rume.time_frame.select.all(),
        rume.ipm.select.compartments("S"),
    )
    existing_fig, _ = plt.subplots()
    try:
        open_figures = plt.get_fignums()

        def fail_transform(data):
            raise RuntimeError("transform failed")

        with pytest.raises(RuntimeError, match="transform failed"):
            filter_output.plot.spaghetti(*args, transform=fail_transform)
        assert plt.get_fignums() == open_figures
    finally:
        plt.close(existing_fig)


##################
# quantile plots #
##################


@pytest.mark.parametrize(
    ("ordering", "expected_groups"),
    [
        ("location", ["AZ: S", "AZ: I", "NM: S", "NM: I"]),
        ("quantity", ["AZ: S", "NM: S", "AZ: I", "NM: I"]),
    ],
)
def test_quantiles_plt_ordering_and_intervals(
    rume, filter_output, ordering, expected_groups
):
    for realization, value in enumerate((0, 10, 20)):
        filter_output.compartments[realization, ...] = value
    transformed = []

    def transform(data):
        transformed.append(data.copy())
        return data.assign(value=data["value"] * 2)

    fig, ax = plt.subplots()
    try:
        lines = filter_output.plot.quantiles_plt(
            ax,
            rume.scope.select.all(),
            rume.time_frame.select.all(),
            rume.ipm.select.compartments("S", "I"),
            credible_intervals=[50, 100],
            ordering=ordering,
            line_kwargs=[{"color": "red"}, {"color": "blue"}],
            fill_kwargs=[{"color": "green"}, {"color": "orange"}],
            transform=transform,
        )
        assert [line.get_label() for line in lines] == [
            f"{group}: Median" for group in expected_groups
        ]
        assert [line.get_color() for line in lines] == ["red", "blue", "red", "blue"]
        assert len(ax.collections) == 8
        for index, collection in enumerate(ax.collections):
            expected_color = "green" if index // 2 % 2 == 0 else "orange"
            np.testing.assert_array_equal(
                np.asarray(collection.get_facecolor())[0, :3],
                to_rgb(expected_color),
            )
            vertices = np.asarray(collection.get_paths()[0].vertices, dtype=float)
            expected_bounds = (0, 40) if index % 2 == 0 else (10, 30)
            np.testing.assert_allclose(
                [vertices[:, 1].min(), vertices[:, 1].max()], expected_bounds
            )
        assert len(transformed) == 20
        assert all(list(data.columns) == ["value"] for data in transformed)
        np.testing.assert_array_equal(lines[0].get_xdata(), [1, 2, 3, 4])
        np.testing.assert_array_equal(lines[0].get_ydata(), [20, 20, 20, 20])
        assert ax.get_legend() is None
        assert ax.get_xlabel() == ""
        assert len(ax.get_legend_handles_labels()[1]) == 8
    finally:
        plt.close(fig)


def test_quantiles_plt_subplots_and_parameters(rume, filter_output):
    fig, axes = plt.subplots(1, 2)
    try:
        for ax, state, expected in zip(axes, ("04", "35"), ("AZ", "NM")):
            lines = filter_output.plot.quantiles_plt(
                ax,
                rume.scope.select.by_state(state),
                rume.time_frame.select.all(),
                filter_output.param_select.by_name("test_0"),
                label_format="{q} @ {n}: {c}",
                median_label_format="{q} @ {n}",
                time_format="date",
            )
            assert [line.get_label() for line in lines] == [
                f"*::*::test_0 @ {expected}: Median"
            ]
            assert ax.get_legend_handles_labels()[1] == [
                f"*::*::test_0 @ {expected}: [95]",
                f"*::*::test_0 @ {expected}: Median",
            ]
            pd.testing.assert_index_equal(
                pd.DatetimeIndex(lines[0].get_xdata()),
                pd.date_range("2021-01-02", periods=4),
            )
    finally:
        plt.close(fig)


def test_quantiles_plt_repeated_and_zero_intervals(rume, filter_output):
    fig, ax = plt.subplots()
    try:
        lines = filter_output.plot.quantiles_plt(
            ax,
            rume.scope.select.by_state("04"),
            rume.time_frame.select.all(),
            rume.ipm.select.compartments("S"),
            credible_intervals=[0, 50, 50],
        )
        assert len(lines) == 1
        assert len(ax.collections) == 3
        np.testing.assert_array_equal(lines[0].get_ydata(), [1, 1, 1, 1])
    finally:
        plt.close(fig)


def test_quantiles_plt_supported_interval_widths(rume, filter_output):
    for realization, value in enumerate((0, 10, 20)):
        filter_output.compartments[realization, ...] = value
    widths = list(range(0, 101, 5))
    fig, ax = plt.subplots()
    try:
        filter_output.plot.quantiles_plt(
            ax,
            rume.scope.select.by_state("04"),
            rume.time_frame.select.all(),
            rume.ipm.select.compartments("S"),
            credible_intervals=widths,
        )
        assert len(ax.collections) == len(widths)
        for collection, width in zip(ax.collections, reversed(widths)):
            vertices = np.asarray(collection.get_paths()[0].vertices, dtype=float)
            np.testing.assert_allclose(
                [vertices[:, 1].min(), vertices[:, 1].max()],
                [10 - width / 10, 10 + width / 10],
            )
    finally:
        plt.close(fig)


def test_quantiles_plt_default_interval_and_styles(rume, filter_output):
    for realization, value in enumerate((0, 10, 20)):
        filter_output.compartments[realization, ...] = value
    fig, ax = plt.subplots()
    try:
        lines = filter_output.plot.quantiles_plt(
            ax,
            rume.scope.select.by_state("04"),
            rume.time_frame.select.all(),
            rume.ipm.select.compartments("S"),
            line_kwargs=[],
            fill_kwargs=[],
        )
        vertices = np.asarray(ax.collections[0].get_paths()[0].vertices, dtype=float)
        np.testing.assert_allclose(
            [vertices[:, 1].min(), vertices[:, 1].max()], [0.5, 19.5]
        )
        assert lines[0].get_color() == "black"
        np.testing.assert_allclose(
            np.asarray(ax.collections[0].get_facecolor())[0],
            (*to_rgb("tab:blue"), 0.3),
        )
    finally:
        plt.close(fig)


def test_quantiles_plt_geography_and_quantity_aggregations(rume, filter_output):
    fig, ax = plt.subplots()
    try:
        lines = filter_output.plot.quantiles_plt(
            ax,
            rume.scope.select.all().sum(),
            rume.time_frame.select.all(),
            rume.ipm.select.compartments("S", "I").group().sum(),
        )
        assert [line.get_label() for line in lines] == ["*: S: Median", "*: I: Median"]
        for line in lines:
            np.testing.assert_array_equal(line.get_ydata(), [2, 2, 2, 2])
    finally:
        plt.close(fig)


@pytest.mark.parametrize("interval", [-5, 2.5, 94.99, 97.5, 105, float("nan")])
def test_quantiles_plt_rejects_unsupported_intervals(rume, filter_output, interval):
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="between 0 and 100 in increments of 5"):
            filter_output.plot.quantiles_plt(
                ax,
                rume.scope.select.by_state("04"),
                rume.time_frame.select.all(),
                rume.ipm.select.compartments("S"),
                credible_intervals=[interval],
            )
    finally:
        plt.close(fig)


@pytest.mark.parametrize("legend", ["auto", "on", "off", "outside"])
def test_quantiles_legend(rume, filter_output, monkeypatch, legend):
    monkeypatch.setattr(plt, "show", lambda: None)
    try:
        filter_output.plot.quantiles(
            rume.scope.select.all(),
            rume.time_frame.select.all(),
            rume.ipm.select.compartments("S", "I"),
            ordering="quantity",
            label_format="{q} @ {n}: {c}",
            median_label_format="{q} @ {n}",
            fill_kwargs=[{"alpha": 0.3}],
            legend=legend,
            time_format="date",
            title="Forecast",
        )
        fig = plt.gcf()
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert ax.get_title() == "Forecast"
        assert ax.get_xlabel() == "date"
        assert ax.get_ylabel() == "count"
        assert [line.get_label() for line in ax.lines] == [
            "S @ AZ: Median",
            "S @ NM: Median",
            "I @ AZ: Median",
            "I @ NM: Median",
        ]
        if legend == "off":
            assert ax.get_legend() is None
        else:
            plot_legend = ax.get_legend()
            assert plot_legend is not None
    finally:
        plt.close()


@pytest.mark.parametrize(("extra_item", "has_legend"), [(False, True), (True, False)])
def test_quantiles_auto_legend_counts_items(
    rume, filter_output, monkeypatch, extra_item, has_legend
):
    renderer = filter_output.plot
    draw_quantiles = renderer.quantiles_plt
    monkeypatch.setattr(plt, "show", lambda: None)

    def draw_with_extra_item(ax, *args, **kwargs):
        lines = draw_quantiles(ax, *args, **kwargs)
        if extra_item:
            ax.plot([0, 1], [0, 1], label="extra")
        return lines

    monkeypatch.setattr(renderer, "quantiles_plt", draw_with_extra_item)
    try:
        renderer.quantiles(
            rume.scope.select.all(),
            rume.time_frame.select.all(),
            rume.ipm.select.compartments("S", "I", "R"),
        )
        ax = plt.gca()
        assert len(ax.get_legend_handles_labels()[1]) == 12 + extra_item
        assert (ax.get_legend() is not None) is has_legend
    finally:
        plt.close()


def test_quantiles_save_single_plot(rume, filter_output, tmp_path):
    path = tmp_path / "quantiles.png"
    try:
        filter_output.plot.quantiles(
            rume.scope.select.all(),
            rume.time_frame.select.all(),
            rume.ipm.select.compartments("S"),
            to_file=path,
        )
        assert path.is_file()
        assert path.stat().st_size > 0
    finally:
        plt.close()


def test_quantiles_closes_figure_on_failure(rume, filter_output):
    existing_fig, _ = plt.subplots()
    try:
        open_figures = plt.get_fignums()

        def fail_transform(data):
            raise RuntimeError("transform failed")

        with pytest.raises(RuntimeError, match="transform failed"):
            filter_output.plot.quantiles(
                rume.scope.select.all(),
                rume.time_frame.select.all(),
                rume.ipm.select.compartments("S"),
                transform=fail_transform,
            )
        assert plt.get_fignums() == open_figures
    finally:
        plt.close(existing_fig)


###################
# histogram plots #
###################


@pytest.mark.parametrize(
    ("ordering", "expected_groups", "expected_values"),
    [
        ("location", ["AZ: S", "AZ: I", "NM: S", "NM: I"], [0, 10, 100, 110]),
        ("quantity", ["AZ: S", "NM: S", "AZ: I", "NM: I"], [0, 100, 10, 110]),
    ],
)
def test_histogram_plt_ordering_and_transform(
    rume, filter_output, ordering, expected_groups, expected_values
):
    for realization in range(3):
        for node in range(2):
            for quantity in range(2):
                filter_output.compartments[realization, :, node, quantity] = (
                    realization + 100 * node + 10 * quantity
                )
    transformed = []

    def transform(data):
        assert list(data.columns) == ["value"]
        transformed.append(data["value"].tolist())
        return data.assign(value=data["value"] + 0.5)

    fig, ax = plt.subplots()
    try:
        plots = filter_output.plot.histogram_plt(
            ax,
            rume.scope.select.all(),
            rume.time_frame.select.days(0, 0),
            rume.ipm.select.compartments("S", "I"),
            ordering=ordering,
            label_format="{n}: {q}",
            hist_kwargs=[
                {"color": "red", "bins": 3},
                {"color": "blue", "bins": 3},
            ],
            transform=transform,
        )
        assert len(plots) == 4
        assert ax.get_legend_handles_labels()[1] == expected_groups
        assert transformed == [[v, v + 1, v + 2] for v in expected_values]
        for plot, value, color in zip(
            plots, expected_values, ("red", "blue", "red", "blue")
        ):
            assert isinstance(plot, BarContainer)
            np.testing.assert_array_equal([bar.get_height() for bar in plot], [1, 1, 1])
            assert plot[0].get_x() == pytest.approx(value + 0.5)
            np.testing.assert_allclose(plot[0].get_facecolor()[:3], to_rgb(color))
        assert ax.get_legend() is None
        assert ax.get_xlabel() == ""
        assert ax.get_title() == ""
    finally:
        plt.close(fig)


def test_histogram_plt_subplots_and_parameters(rume, filter_output):
    fig, axes = plt.subplots(1, 2)
    try:
        for ax, state, expected in zip(axes, ("04", "35"), ("AZ", "NM")):
            plots = filter_output.plot.histogram_plt(
                ax,
                rume.scope.select.by_state(state),
                rume.time_frame.select.days(0, 0),
                filter_output.param_select.by_name("test_0"),
                label_format="{q} @ {n}: {t}",
                hist_kwargs=[{"bins": 2}],
            )
            assert len(plots) == 1
            assert isinstance(plots[0], BarContainer)
            assert ax.get_legend_handles_labels()[1] == [
                f"*::*::test_0 @ {expected}: 2021-01-01"
            ]
            assert sum(bar.get_height() for bar in plots[0]) == 3
    finally:
        plt.close(fig)


def test_histogram_plt_step_histogram(rume, filter_output):
    fig, ax = plt.subplots()
    try:
        plots = filter_output.plot.histogram_plt(
            ax,
            rume.scope.select.by_state("04"),
            rume.time_frame.select.days(0, 0),
            rume.ipm.select.compartments("S"),
            hist_kwargs=[{"histtype": "step"}],
        )
        assert len(plots) == 1
        assert isinstance(plots[0], list)
        assert len(plots[0]) == 1
        assert plots[0][0] in ax.patches
    finally:
        plt.close(fig)


def test_histogram_plt_mixed_histogram_types(rume, filter_output):
    fig, ax = plt.subplots()
    try:
        plots = filter_output.plot.histogram_plt(
            ax,
            rume.scope.select.all(),
            rume.time_frame.select.days(0, 0),
            rume.ipm.select.compartments("S", "I"),
            hist_kwargs=[
                {"histtype": "bar"},
                {"histtype": "step"},
                {"histtype": "barstacked"},
                {"histtype": "stepfilled"},
            ],
        )
        assert len(plots) == 4
        assert isinstance(plots[0], BarContainer)
        assert isinstance(plots[1], list)
        assert len(plots[1]) == 1
        assert isinstance(plots[1][0], Polygon)
        assert isinstance(plots[2], BarContainer)
        assert isinstance(plots[3], list)
        assert len(plots[3]) == 1
        assert isinstance(plots[3][0], Polygon)
    finally:
        plt.close(fig)


def test_histogram_plt_rejects_unexpected_artists(rume, filter_output, monkeypatch):
    fig, ax = plt.subplots()
    monkeypatch.setattr(
        ax, "hist", lambda *args, **kwargs: ([], [], [BarContainer([])])
    )
    try:
        with pytest.raises(TypeError, match="unexpected histogram artists"):
            filter_output.plot.histogram_plt(
                ax,
                rume.scope.select.by_state("04"),
                rume.time_frame.select.days(0, 0),
                rume.ipm.select.compartments("S"),
            )
    finally:
        plt.close(fig)


def test_histogram_plt_aggregated_selections(rume, filter_output):
    fig, ax = plt.subplots()
    try:
        plots = filter_output.plot.histogram_plt(
            ax,
            rume.scope.select.all().sum(),
            rume.time_frame.select.all().agg(),
            rume.ipm.select.compartments("S", "I").group().sum(),
            label_format="{n}: {q}: {t}",
            hist_kwargs=[],
        )
        assert len(plots) == 2
        assert ax.get_legend_handles_labels()[1] == [
            "*: S: 2021-01-01/2021-01-04",
            "*: I: 2021-01-01/2021-01-04",
        ]
        for plot in plots:
            assert isinstance(plot, BarContainer)
            assert sum(bar.get_height() for bar in plot) == 3
            bar = next(bar for bar in plot if bar.get_height() > 0)
            assert bar.get_x() <= 2 <= bar.get_x() + bar.get_width()
    finally:
        plt.close(fig)


def test_histogram_plt_rejects_multiple_times(rume, filter_output):
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="single point"):
            filter_output.plot.histogram_plt(
                ax,
                rume.scope.select.all(),
                rume.time_frame.select.all(),
                rume.ipm.select.compartments("S"),
            )
        assert len(ax.patches) == 0
    finally:
        plt.close(fig)


@pytest.mark.parametrize("legend", ["auto", "on", "off", "outside"])
def test_histogram_legend(rume, filter_output, monkeypatch, legend):
    monkeypatch.setattr(plt, "show", lambda: None)
    try:
        filter_output.plot.histogram(
            rume.scope.select.all(),
            rume.time_frame.select.days(0, 0),
            rume.ipm.select.compartments("S", "I"),
            label_format="{q} @ {n}",
            ordering="quantity",
            legend=legend,
            title="Forecast",
        )
        fig = plt.gcf()
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert ax.get_title() == "Forecast"
        assert ax.get_xlabel() == "value"
        assert ax.get_ylabel() == "count"
        assert ax.get_legend_handles_labels()[1] == [
            "S @ AZ",
            "S @ NM",
            "I @ AZ",
            "I @ NM",
        ]
        assert (ax.get_legend() is not None) is (legend != "off")
    finally:
        plt.close()


@pytest.mark.parametrize(("groups", "has_legend"), [(12, True), (13, False)])
def test_histogram_auto_legend_counts_groups(
    rume, filter_output, monkeypatch, groups, has_legend
):
    renderer = filter_output.plot
    monkeypatch.setattr(plt, "show", lambda: None)

    def draw_groups(ax, *args, **kwargs):
        for group in range(groups):
            ax.hist([0, 1], label=f"group {group}")
        return []

    monkeypatch.setattr(renderer, "histogram_plt", draw_groups)
    try:
        renderer.histogram(
            rume.scope.select.all(),
            rume.time_frame.select.days(0, 0),
            rume.ipm.select.compartments("S"),
        )
        assert (plt.gca().get_legend() is not None) is has_legend
    finally:
        plt.close()


def test_histogram_save_single_plot(rume, filter_output, tmp_path):
    path = tmp_path / "histogram.png"
    try:
        filter_output.plot.histogram(
            rume.scope.select.all(),
            rume.time_frame.select.days(0, 0),
            rume.ipm.select.compartments("S"),
            to_file=path,
        )
        assert path.is_file()
        assert path.stat().st_size > 0
    finally:
        plt.close()


def test_histogram_closes_figure_on_failure(rume, filter_output):
    existing_fig, _ = plt.subplots()
    try:
        open_figures = plt.get_fignums()

        def fail_transform(data):
            raise RuntimeError("transform failed")

        with pytest.raises(RuntimeError, match="transform failed"):
            filter_output.plot.histogram(
                rume.scope.select.all(),
                rume.time_frame.select.days(0, 0),
                rume.ipm.select.compartments("S"),
                transform=fail_transform,
            )
        assert plt.get_fignums() == open_figures
    finally:
        plt.close(existing_fig)


#############
# KDE plots #
#############


@pytest.mark.parametrize(
    ("ordering", "expected_groups", "expected_values"),
    [
        ("location", ["AZ: S", "AZ: I", "NM: S", "NM: I"], [1, 2, 3, 4]),
        ("quantity", ["AZ: S", "NM: S", "AZ: I", "NM: I"], [1, 3, 2, 4]),
    ],
)
def test_kde_plt_ordering_and_transform(
    rume, filter_output, ordering, expected_groups, expected_values
):
    for realization in range(3):
        for node in range(2):
            for quantity in range(2):
                scale = 1 + 2 * node + quantity
                filter_output.compartments[realization, :, node, quantity] = (
                    realization * scale
                )
    transformed = []

    def transform(data):
        assert list(data.columns) == ["value"]
        transformed.append(data["value"].tolist())
        return data.assign(value=data["value"] + 0.5)

    fig, ax = plt.subplots()
    try:
        lines = filter_output.plot.kde_plt(
            ax,
            rume.scope.select.all(),
            rume.time_frame.select.days(0, 0),
            rume.ipm.select.compartments("S", "I"),
            label_format="{n}: {q}",
            ordering=ordering,
            line_kwargs=[{"color": "red"}, {"color": "blue"}],
            bandwidth="silverman",
            transform=transform,
        )
        assert list(ax.lines) == lines
        assert [line.get_label() for line in lines] == expected_groups
        assert [line.get_color() for line in lines] == ["red", "blue", "red", "blue"]
        assert transformed == [[0, scale, 2 * scale] for scale in expected_values]
        for line, scale in zip(lines, expected_values):
            x = np.asarray(line.get_xdata())
            expected_x = np.arange(0.5, 2 * scale + 0.5 + scale / 50, scale / 50)
            np.testing.assert_allclose(x, expected_x)
            expected_density = gaussian_kde(
                [0.5, scale + 0.5, 2 * scale + 0.5], bw_method="silverman"
            )(expected_x)
            np.testing.assert_allclose(line.get_ydata(), expected_density)
        assert ax.get_legend() is None
        assert ax.get_xlabel() == ""
        assert ax.get_title() == ""
    finally:
        plt.close(fig)


def test_kde_plt_subplots_and_parameters(rume, filter_output):
    values = next(iter(filter_output.estimated_params.values()))
    values[:, :, :] = np.arange(3)[:, np.newaxis, np.newaxis]
    fig, axes = plt.subplots(1, 2)
    try:
        for ax, state, expected in zip(axes, ("04", "35"), ("AZ", "NM")):
            lines = filter_output.plot.kde_plt(
                ax,
                rume.scope.select.by_state(state),
                rume.time_frame.select.days(0, 0),
                filter_output.param_select.by_name("test_0"),
                label_format="{q} @ {n}: {t}",
                delta_t=0.5,
                bandwidth=0.5,
                line_kwargs=[],
            )
            assert len(lines) == 1
            assert lines[0].get_label() == (f"*::*::test_0 @ {expected}: 2021-01-01")
            np.testing.assert_allclose(lines[0].get_xdata(), [0, 0.5, 1, 1.5, 2])
            np.testing.assert_allclose(
                lines[0].get_ydata(),
                gaussian_kde([0, 1, 2], bw_method=0.5)([0, 0.5, 1, 1.5, 2]),
            )
    finally:
        plt.close(fig)


def test_kde_plt_aggregated_selections(rume, filter_output):
    for realization in range(3):
        filter_output.compartments[realization, ...] = realization
    fig, ax = plt.subplots()
    try:
        lines = filter_output.plot.kde_plt(
            ax,
            rume.scope.select.all().sum(),
            rume.time_frame.select.all().agg(),
            rume.ipm.select.compartments("S", "I").group().sum(),
            delta_t=1,
        )
        assert [line.get_label() for line in lines] == [
            "*: S: 2021-01-01/2021-01-04",
            "*: I: 2021-01-01/2021-01-04",
        ]
        for line in lines:
            np.testing.assert_array_equal(line.get_xdata(), [0, 1, 2, 3, 4])
            np.testing.assert_allclose(
                line.get_ydata(), gaussian_kde([0, 2, 4])([0, 1, 2, 3, 4])
            )
    finally:
        plt.close(fig)


def test_kde_plt_rejects_constant_data(rume, filter_output):
    fig, ax = plt.subplots()
    try:
        with pytest.raises(np.linalg.LinAlgError, match="singular"):
            filter_output.plot.kde_plt(
                ax,
                rume.scope.select.by_state("04"),
                rume.time_frame.select.days(0, 0),
                rume.ipm.select.compartments("S"),
            )
    finally:
        plt.close(fig)


def test_kde_plt_rejects_multiple_times(rume, filter_output):
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="single point"):
            filter_output.plot.kde_plt(
                ax,
                rume.scope.select.all(),
                rume.time_frame.select.all(),
                rume.ipm.select.compartments("S"),
            )
        assert len(ax.lines) == 0
    finally:
        plt.close(fig)


@pytest.mark.parametrize("delta_t", [0, -0.5, float("nan"), float("inf")])
def test_kde_plt_rejects_invalid_delta_t(rume, filter_output, delta_t):
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="delta_t"):
            filter_output.plot.kde_plt(
                ax,
                rume.scope.select.all(),
                rume.time_frame.select.days(0, 0),
                rume.ipm.select.compartments("S"),
                delta_t=delta_t,
            )
    finally:
        plt.close(fig)


@pytest.mark.parametrize("legend", ["auto", "on", "off", "outside"])
def test_kde_legend(rume, filter_output, monkeypatch, legend):
    for realization in range(3):
        filter_output.compartments[realization, ...] = realization
    monkeypatch.setattr(plt, "show", lambda: None)
    try:
        filter_output.plot.kde(
            rume.scope.select.all(),
            rume.time_frame.select.days(0, 0),
            rume.ipm.select.compartments("S", "I"),
            label_format="{q} @ {n}",
            ordering="quantity",
            line_kwargs=[{"alpha": 0.3}],
            legend=legend,
            title="Forecast",
        )
        fig = plt.gcf()
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert ax.get_title() == "Forecast"
        assert ax.get_xlabel() == "value"
        assert ax.get_ylabel() == "density"
        assert ax.get_legend_handles_labels()[1] == [
            "S @ AZ",
            "S @ NM",
            "I @ AZ",
            "I @ NM",
        ]
        plot_legend = ax.get_legend()
        assert (plot_legend is not None) is (legend != "off")
        if plot_legend is not None:
            assert all(
                handle is not None and handle.get_alpha() == 0.3
                for handle in plot_legend.legend_handles
            )
    finally:
        plt.close()


@pytest.mark.parametrize(("groups", "has_legend"), [(12, True), (13, False)])
def test_kde_auto_legend_counts_groups(
    rume, filter_output, monkeypatch, groups, has_legend
):
    renderer = filter_output.plot
    monkeypatch.setattr(plt, "show", lambda: None)

    def draw_groups(ax, *args, **kwargs):
        return [
            ax.plot([0, 1], [0, 1], label=f"group {group}")[0]
            for group in range(groups)
        ]

    monkeypatch.setattr(renderer, "kde_plt", draw_groups)
    try:
        renderer.kde(
            rume.scope.select.all(),
            rume.time_frame.select.days(0, 0),
            rume.ipm.select.compartments("S"),
        )
        assert (plt.gca().get_legend() is not None) is has_legend
    finally:
        plt.close()


def test_kde_save_single_plot(rume, filter_output, tmp_path):
    for realization in range(3):
        filter_output.compartments[realization, ...] = realization
    path = tmp_path / "kde.png"
    try:
        filter_output.plot.kde(
            rume.scope.select.all(),
            rume.time_frame.select.days(0, 0),
            rume.ipm.select.compartments("S"),
            to_file=path,
        )
        assert path.is_file()
        assert path.stat().st_size > 0
    finally:
        plt.close()


def test_kde_closes_figure_on_failure(rume, filter_output):
    existing_fig, _ = plt.subplots()
    try:
        open_figures = plt.get_fignums()

        def fail_transform(data):
            raise RuntimeError("transform failed")

        with pytest.raises(RuntimeError, match="transform failed"):
            filter_output.plot.kde(
                rume.scope.select.all(),
                rume.time_frame.select.days(0, 0),
                rume.ipm.select.compartments("S"),
                transform=fail_transform,
            )
        assert plt.get_fignums() == open_figures
    finally:
        plt.close(existing_fig)
