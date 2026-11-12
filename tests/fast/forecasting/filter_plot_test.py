from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgb
from matplotlib.container import BarContainer
from matplotlib.patches import Polygon
from scipy.stats import gaussian_kde


@pytest.fixture(autouse=True)
def no_plot_show(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Suppress plot display and close all figures after the test."""
    monkeypatch.setattr(plt, "show", lambda: None)
    yield
    plt.close("all")


@pytest.mark.parametrize(
    ("method", "expected_values"),
    [
        ("line_selection_plt", [[3, 9], [103, 109], [203, 209]]),
        ("line_plt", [[103, 109]]),
        ("quantiles_plt", [[103, 109]]),
    ],
)
def test_plot_preserves_epiweek_order(temporal_order_output, method, expected_values):
    output = temporal_order_output
    rume = output.rume
    geo = rume.scope.select.by_state("04")
    time = rume.time_frame.select.all().group("epiweek").agg()
    quantity = rume.ipm.select.compartments("S")
    _, ax = plt.subplots()
    if method in ("line_plt", "line_selection_plt"):
        realization = output.select_realization.all()
        if method == "line_plt":
            realization = realization.mean()
        lines = output.plot.line_plt(
            ax,
            realization=realization,
            geo=geo,
            time=time,
            quantity=quantity,
        )
    else:
        lines = output.plot.quantiles_plt(
            ax,
            geo=geo,
            time=time,
            quantity=quantity,
        )

    assert len(lines) == len(expected_values)
    for line, values in zip(lines, expected_values, strict=True):
        np.testing.assert_array_equal(line.get_xdata(), ["2021-9", "2021-10"])
        np.testing.assert_array_equal(line.get_ydata(), values)


@pytest.mark.parametrize(
    ("ordering", "expected_groups"),
    [
        ("location", ["S @ AZ", "I @ AZ", "S @ NM", "I @ NM"]),
        ("quantity", ["S @ AZ", "S @ NM", "I @ AZ", "I @ NM"]),
    ],
)
@pytest.mark.parametrize(
    ("method", "label_format", "label_suffix"),
    [
        ("line", "{q} @ {n}: {m}", ": mean"),
        ("quantiles", "{q} @ {n}: {c}", ": Median"),
        ("histogram", "{q} @ {n}", ""),
        ("kernel_density", "{q} @ {n}", ""),
    ],
)
def test_plot_ordering(
    rume,
    filter_output,
    ordering,
    expected_groups,
    method,
    label_format,
    label_suffix,
):
    for realization in range(3):
        filter_output.compartments[realization, ...] = realization
    geo = rume.scope.select.all()
    time = (
        rume.time_frame.select.days(0, 0)
        if method in ("histogram", "kernel_density")
        else rume.time_frame.select.all()
    )
    quantity = rume.ipm.select.compartments("S", "I")
    if method == "line":
        filter_output.plot.line(
            realization=filter_output.select_realization.all().mean(),
            geo=geo,
            time=time,
            quantity=quantity,
            ordering=ordering,
            label_format=label_format,
        )
    else:
        plot_method = getattr(filter_output.plot, method)
        plot_method(
            geo=geo,
            time=time,
            quantity=quantity,
            ordering=ordering,
            label_format=label_format,
        )

    ax = plt.gca()
    if method == "quantiles":
        labels = [line.get_label() for line in ax.lines]
    else:
        labels = ax.get_legend_handles_labels()[1]
    assert labels == [f"{group}{label_suffix}" for group in expected_groups]


##############
# line plots #
##############


def test_line_plt_basic(rume, filter_output):
    _, ax = plt.subplots()
    lines = filter_output.plot.line_plt(
        ax,
        realization=filter_output.select_realization.all().agg(["mean"]),
        geo=rume.scope.select.by_state("04"),
        time=rume.time_frame.select.all(),
        quantity=rume.ipm.select.compartments("S"),
    )
    assert len(lines) == 1
    assert lines[0].get_label() == "AZ S mean"
    np.testing.assert_array_equal(lines[0].get_xdata(), [1, 2, 3, 4])
    np.testing.assert_array_equal(lines[0].get_ydata(), [1, 1, 1, 1])


@pytest.mark.parametrize("aggregate", [False, True])
def test_line_plt_transform_columns(rume, filter_output, aggregate):
    seen = []

    def transform(data):
        seen.append(list(data.columns))
        return data

    if aggregate:
        realization = filter_output.select_realization.all().mean()
        expected_columns = ["time", "geo", "quantity", "metric", "value"]
        expected_calls = 1
    else:
        realization = filter_output.select_realization.all()
        expected_columns = ["realization", "time", "geo", "quantity", "value"]
        expected_calls = 3

    _, ax = plt.subplots()
    filter_output.plot.line_plt(
        ax,
        realization=realization,
        geo=rume.scope.select.by_state("04"),
        time=rume.time_frame.select.all(),
        quantity=rume.ipm.select.compartments("S"),
        transform=transform,
    )
    assert seen == [expected_columns] * expected_calls


@pytest.mark.parametrize("aggregate", [False, True])
def test_line_plt_transform_values(rume, filter_output, aggregate):
    for index, value in enumerate((1, 2, 3)):
        filter_output.compartments[index, ...] = value
    if aggregate:
        realization = filter_output.select_realization.all().mean()
        expected_values = [4]
    else:
        realization = filter_output.select_realization.all()
        expected_values = [2, 4, 6]

    _, ax = plt.subplots()
    lines = filter_output.plot.line_plt(
        ax,
        realization=realization,
        geo=rume.scope.select.by_state("04"),
        time=rume.time_frame.select.all(),
        quantity=rume.ipm.select.compartments("S"),
        transform=lambda data: data.assign(value=data["value"] * 2),
    )
    assert len(lines) == len(expected_values)
    for line, value in zip(lines, expected_values, strict=True):
        np.testing.assert_array_equal(line.get_ydata(), [value] * 4)


def test_line_color_cycle(rume, filter_output):
    filter_output.plot.line(
        realization=filter_output.select_realization.all().mean(),
        geo=rume.scope.select.all(),
        time=rume.time_frame.select.all(),
        quantity=rume.ipm.select.compartments("S", "I"),
        line_kwargs=[{"color": "red"}, {"color": "blue"}],
    )
    assert [line.get_color() for line in plt.gca().lines] == [
        "red",
        "blue",
        "red",
        "blue",
    ]


def test_line_plt_parameter_aggregation(rume, filter_output):
    values = next(iter(filter_output.estimated_params.values()))
    values[:, :, :] = np.array([0, 20, 40])[:, np.newaxis, np.newaxis]
    _, ax = plt.subplots()
    lines = filter_output.plot.line_plt(
        ax,
        realization=filter_output.select_realization.all().agg(
            ["quantile_2.5", "mean"]
        ),
        geo=filter_output.rume.scope.select.all().sum(),
        time=filter_output.rume.time_frame.select.all(),
        quantity=filter_output.select_parameter.by_name("test_0"),
        label_format="{q} @ {n}: {m}",
        line_kwargs=[],
    )
    assert [line.get_label() for line in lines] == [
        "*::*::test_0 @ *: quantile_2.5",
        "*::*::test_0 @ *: mean",
    ]
    np.testing.assert_allclose(lines[0].get_ydata(), [2, 2, 2, 2])
    np.testing.assert_array_equal(lines[1].get_ydata(), [40, 40, 40, 40])


def test_line_plt_date_format(rume, filter_output):
    _, ax = plt.subplots()
    lines = filter_output.plot.line_plt(
        ax,
        realization=filter_output.select_realization.all().mean(),
        geo=filter_output.rume.scope.select.by_state("04"),
        time=filter_output.rume.time_frame.select.all(),
        quantity=filter_output.rume.ipm.select.compartments("S"),
        time_format="date",
    )
    pd.testing.assert_index_equal(
        pd.DatetimeIndex(lines[0].get_xdata()),
        pd.date_range("2021-01-02", periods=4),
    )


@pytest.mark.parametrize("aggregate", [False, True])
def test_line_plt_grouped_quantities(multistrata_filter_output, aggregate):
    output = multistrata_filter_output
    rume = output.rume
    for name, value in (("S_young", 1), ("S_old", 4), ("I_young", 2), ("I_old", 8)):
        index = rume.ipm.select.compartments(name).compartment_index
        output.compartments[..., index] = (
            value + np.arange(output.num_realizations)[:, np.newaxis, np.newaxis]
        )

    if aggregate:
        realization = output.select_realization.all().mean()
        expected_values = ([7], [12])
        expected_labels = ["AZ S mean", "AZ I mean"]
    else:
        realization = output.select_realization.all()
        expected_values = ([5, 7, 9], [10, 12, 14])
        expected_labels = ["AZ S", "AZ I"]

    _, ax = plt.subplots()
    lines = output.plot.line_plt(
        ax,
        realization=realization,
        geo=rume.scope.select.by_state("04"),
        time=rume.time_frame.select.all(),
        quantity=rume.ipm.select.compartments("S_*", "I_*").group(strata=True).sum(),
    )
    assert len(lines) == sum(len(values) for values in expected_values)
    lines_per_group = len(expected_values[0])
    assert [line.get_label() for line in lines[::lines_per_group]] == expected_labels
    for line, value in zip(
        lines, (*expected_values[0], *expected_values[1]), strict=True
    ):
        np.testing.assert_array_equal(line.get_ydata(), [value] * 4)


@pytest.mark.parametrize("aggregate", [False, True])
def test_line_plt_duplicate_labels(temporal_order_output, monkeypatch, aggregate):
    # Make sure that the disambiguation of labels works correctly when the same label
    # is used for multiple series. Check that the series are kept separate.
    # This test is repeated with aggregation and without.
    out = temporal_order_output
    quantity = out.rume.ipm.select.compartments("S", "I")

    # Monkeypatch to force the labels to turn out the same.
    monkeypatch.setattr(
        type(quantity),
        "disambiguate_groups",
        lambda self: {"S_0": "same", "I_1": "same"},
    )

    if aggregate:
        realization = out.select_realization.all().mean()
        lines_per_group = 1
        labels_per_group = ["AZ same mean"]
        data_offset = 100
    else:
        realization = out.select_realization.all()
        lines_per_group = 3
        labels_per_group = ["AZ same", "_nolegend_", "_nolegend_"]
        data_offset = 0

    _, ax = plt.subplots()
    lines = out.plot.line_plt(
        ax,
        realization=realization,
        geo=out.rume.scope.select.by_state("04"),
        time=out.rume.time_frame.select.all(),
        quantity=quantity,
        # also check that user-supplied labels are ignored
        line_kwargs=[{"label": "IGNORED"}],
    )

    assert len(lines) == 2 * lines_per_group
    assert [line.get_label() for line in lines] == labels_per_group * 2
    np.testing.assert_array_equal(
        lines[0].get_ydata(),
        np.array([0, 3, 6, 9]) + data_offset,
    )
    np.testing.assert_array_equal(
        lines[lines_per_group].get_ydata(),
        np.array([1, 4, 7, 10]) + data_offset,
    )


def test_line_plt_realization_selection_subplots_and_parameters(rume, filter_output):
    rng = np.random.default_rng(42)
    _, axes = plt.subplots(1, 2)
    for ax, state, expected in zip(axes, ("04", "35"), ("AZ", "NM")):
        lines = filter_output.plot.line_plt(
            ax,
            realization=filter_output.select_realization.sample(rng=rng, size=1),
            geo=rume.scope.select.by_state(state),
            time=rume.time_frame.select.all(),
            quantity=filter_output.select_parameter.by_name("test_0"),
        )
        assert len(lines) == 1
        assert lines[0].get_label() == f"{expected} *::*::test_0"


@pytest.mark.parametrize("legend", ["auto", "on", "off", "outside"])
def test_line_realization_selection_legend(rume, filter_output, legend):
    filter_output.plot.line(
        realization=filter_output.select_realization.all(),
        geo=rume.scope.select.all(),
        time=rume.time_frame.select.all(),
        quantity=rume.ipm.select.compartments("S", "I"),
        legend=legend,
    )
    assert (plt.gca().get_legend() is not None) is (legend != "off")


def test_line_plot_labels(rume, filter_output):
    filter_output.plot.line(
        realization=filter_output.select_realization.all(),
        geo=rume.scope.select.by_state("04"),
        time=rume.time_frame.select.all(),
        quantity=rume.ipm.select.compartments("S"),
        time_format="date",
        title="Forecast",
    )
    ax = plt.gca()
    assert ax.get_title() == "Forecast"
    assert ax.get_xlabel() == "date"
    assert ax.get_ylabel() == "count"


##################
# quantile plots #
##################


def test_quantiles_plt_group_styles(rume, filter_output):
    _, ax = plt.subplots()
    lines = filter_output.plot.quantiles_plt(
        ax,
        geo=rume.scope.select.all(),
        time=rume.time_frame.select.all(),
        quantity=rume.ipm.select.compartments("S", "I"),
        credible_intervals=[50, 100],
        line_kwargs=[{"color": "red"}, {"color": "blue"}],
        fill_kwargs=[{"color": "green"}, {"color": "orange"}],
    )
    assert [line.get_color() for line in lines] == ["red", "blue", "red", "blue"]
    assert len(ax.collections) == 8
    for collection, color in zip(
        ax.collections,
        ["green", "green", "orange", "orange"] * 2,
        strict=True,
    ):
        np.testing.assert_array_equal(
            np.asarray(collection.get_facecolor())[0, :3],
            to_rgb(color),
        )


def test_quantiles_plt_transform_columns(rume, filter_output):
    seen = []

    def transform(data):
        seen.append(list(data.columns))
        return data

    _, ax = plt.subplots()
    filter_output.plot.quantiles_plt(
        ax,
        geo=rume.scope.select.by_state("04"),
        time=rume.time_frame.select.all(),
        quantity=rume.ipm.select.compartments("S"),
        transform=transform,
    )
    assert seen == [["time", "geo", "quantity", "quantile", "value"]] * 3


def test_quantiles_plt_transform_values(rume, filter_output):
    for realization, value in enumerate((0, 10, 20)):
        filter_output.compartments[realization, ...] = value

    _, ax = plt.subplots()
    lines = filter_output.plot.quantiles_plt(
        ax,
        geo=rume.scope.select.by_state("04"),
        time=rume.time_frame.select.all(),
        quantity=rume.ipm.select.compartments("S"),
        credible_intervals=[50, 100],
        transform=lambda data: data.assign(value=data["value"] * 2),
    )
    assert len(ax.collections) == 2
    for collection, bounds in zip(ax.collections, [(0, 40), (10, 30)], strict=True):
        vertices = np.asarray(collection.get_paths()[0].vertices, dtype=float)
        np.testing.assert_allclose([vertices[:, 1].min(), vertices[:, 1].max()], bounds)
    np.testing.assert_array_equal(lines[0].get_ydata(), [20, 20, 20, 20])


def test_quantiles_plt_subplots_and_parameters(rume, filter_output):
    _, axes = plt.subplots(1, 2)
    for ax, state, expected in zip(axes, ("04", "35"), ("AZ", "NM")):
        lines = filter_output.plot.quantiles_plt(
            ax,
            geo=rume.scope.select.by_state(state),
            time=rume.time_frame.select.all(),
            quantity=filter_output.select_parameter.by_name("test_0"),
            label_format="{q} @ {n}: {c}",
        )
        assert [line.get_label() for line in lines] == [
            f"*::*::test_0 @ {expected}: Median"
        ]
        assert ax.get_legend_handles_labels()[1] == [
            f"*::*::test_0 @ {expected}: [95]",
            f"*::*::test_0 @ {expected}: Median",
        ]


def test_quantiles_plt_date_format(rume, filter_output):
    _, ax = plt.subplots()
    lines = filter_output.plot.quantiles_plt(
        ax,
        geo=rume.scope.select.by_state("04"),
        time=rume.time_frame.select.all(),
        quantity=rume.ipm.select.compartments("S"),
        time_format="date",
    )
    pd.testing.assert_index_equal(
        pd.DatetimeIndex(lines[0].get_xdata()),
        pd.date_range("2021-01-02", periods=4),
    )


def test_quantiles_plt_repeated_intervals(rume, filter_output):
    _, ax = plt.subplots()
    filter_output.plot.quantiles_plt(
        ax,
        geo=rume.scope.select.by_state("04"),
        time=rume.time_frame.select.all(),
        quantity=rume.ipm.select.compartments("S"),
        credible_intervals=[50, 50],
    )
    assert len(ax.collections) == 2


def test_quantiles_plt_supported_interval_widths(rume, filter_output):
    for realization, value in enumerate((0, 10, 20)):
        filter_output.compartments[realization, ...] = value
    widths = list(range(0, 101, 5))
    _, ax = plt.subplots()
    filter_output.plot.quantiles_plt(
        ax,
        geo=rume.scope.select.by_state("04"),
        time=rume.time_frame.select.all(),
        quantity=rume.ipm.select.compartments("S"),
        credible_intervals=widths,
    )
    assert len(ax.collections) == len(widths)
    for collection, width in zip(ax.collections, reversed(widths)):
        vertices = np.asarray(collection.get_paths()[0].vertices, dtype=float)
        np.testing.assert_allclose(
            [vertices[:, 1].min(), vertices[:, 1].max()],
            [10 - width / 10, 10 + width / 10],
        )


def test_quantiles_plt_default_interval(rume, filter_output):
    for realization, value in enumerate((0, 10, 20)):
        filter_output.compartments[realization, ...] = value
    _, ax = plt.subplots()
    filter_output.plot.quantiles_plt(
        ax,
        geo=rume.scope.select.by_state("04"),
        time=rume.time_frame.select.all(),
        quantity=rume.ipm.select.compartments("S"),
    )
    vertices = np.asarray(ax.collections[0].get_paths()[0].vertices, dtype=float)
    np.testing.assert_allclose(
        [vertices[:, 1].min(), vertices[:, 1].max()],
        [0.5, 19.5],
    )


def test_quantiles_plt_default_styles(rume, filter_output):
    _, ax = plt.subplots()
    lines = filter_output.plot.quantiles_plt(
        ax,
        geo=rume.scope.select.by_state("04"),
        time=rume.time_frame.select.all(),
        quantity=rume.ipm.select.compartments("S"),
        line_kwargs=[],
        fill_kwargs=[],
    )
    assert lines[0].get_color() == "black"
    np.testing.assert_allclose(
        np.asarray(ax.collections[0].get_facecolor())[0],
        (*to_rgb("tab:blue"), 0.3),
    )


def test_quantiles_plt_geography_aggregation(rume, filter_output):
    filter_output.compartments[..., 1] = 3
    _, ax = plt.subplots()
    lines = filter_output.plot.quantiles_plt(
        ax,
        geo=rume.scope.select.all().sum(),
        time=rume.time_frame.select.all(),
        quantity=rume.ipm.select.compartments("S", "I"),
    )
    assert [line.get_label() for line in lines] == ["*: S: Median", "*: I: Median"]
    np.testing.assert_array_equal(lines[0].get_ydata(), [2, 2, 2, 2])
    np.testing.assert_array_equal(lines[1].get_ydata(), [6, 6, 6, 6])


def test_quantiles_plt_grouped_quantities(multistrata_filter_output):
    output = multistrata_filter_output
    rume = output.rume
    for name, value in (("S_young", 1), ("S_old", 4), ("I_young", 2), ("I_old", 8)):
        index = rume.ipm.select.compartments(name).compartment_index
        output.compartments[..., index] = value

    _, ax = plt.subplots()
    lines = output.plot.quantiles_plt(
        ax,
        geo=rume.scope.select.by_state("04"),
        time=rume.time_frame.select.all(),
        quantity=rume.ipm.select.compartments("S_*", "I_*").group(strata=True).sum(),
    )
    assert [line.get_label() for line in lines] == ["AZ: S: Median", "AZ: I: Median"]
    np.testing.assert_array_equal(lines[0].get_ydata(), [5, 5, 5, 5])
    np.testing.assert_array_equal(lines[1].get_ydata(), [10, 10, 10, 10])


@pytest.mark.parametrize("interval", [-5, 2.5, 94.99, 97.5, 105, float("nan")])
def test_quantiles_plt_rejects_unsupported_intervals(rume, filter_output, interval):
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match="between 0 and 100 in increments of 5"):
        filter_output.plot.quantiles_plt(
            ax,
            geo=rume.scope.select.by_state("04"),
            time=rume.time_frame.select.all(),
            quantity=rume.ipm.select.compartments("S"),
            credible_intervals=[interval],
        )


@pytest.mark.parametrize("legend", ["auto", "on", "off", "outside"])
def test_quantiles_legend(rume, filter_output, legend):
    filter_output.plot.quantiles(
        geo=rume.scope.select.all(),
        time=rume.time_frame.select.all(),
        quantity=rume.ipm.select.compartments("S", "I"),
        legend=legend,
    )
    assert (plt.gca().get_legend() is not None) is (legend != "off")


def test_quantiles_plot_labels(rume, filter_output):
    filter_output.plot.quantiles(
        geo=rume.scope.select.by_state("04"),
        time=rume.time_frame.select.all(),
        quantity=rume.ipm.select.compartments("S"),
        time_format="date",
        title="Forecast",
    )
    ax = plt.gca()
    assert ax.get_title() == "Forecast"
    assert ax.get_xlabel() == "date"
    assert ax.get_ylabel() == "count"


###################
# histogram plots #
###################


def test_histogram_plt_transform_columns(rume, filter_output):
    seen = []

    def transform(data):
        seen.append(list(data.columns))
        return data

    _, ax = plt.subplots()
    filter_output.plot.histogram_plt(
        ax,
        geo=rume.scope.select.by_state("04"),
        time=rume.time_frame.select.days(0, 0),
        quantity=rume.ipm.select.compartments("S"),
        transform=transform,
    )
    assert seen == [["geo", "quantity", "value"]]


def test_histogram_plt_transform_values(rume, filter_output):
    for realization, value in enumerate((0, 1, 2)):
        filter_output.compartments[realization, ...] = value
    _, ax = plt.subplots()
    plots = filter_output.plot.histogram_plt(
        ax,
        geo=rume.scope.select.by_state("04"),
        time=rume.time_frame.select.days(0, 0),
        quantity=rume.ipm.select.compartments("S"),
        hist_kwargs=[{"bins": 3}],
        transform=lambda data: data.assign(value=data["value"] + 0.5),
    )
    plot = plots[0]
    assert isinstance(plot, BarContainer)
    np.testing.assert_array_equal([bar.get_height() for bar in plot], [1, 1, 1])
    assert plot[0].get_x() == pytest.approx(0.5)
    assert plot[-1].get_x() + plot[-1].get_width() == pytest.approx(2.5)


def test_histogram_plt_subplots_and_parameters(rume, filter_output):
    _, axes = plt.subplots(1, 2)
    for ax, state, expected in zip(axes, ("04", "35"), ("AZ", "NM")):
        plots = filter_output.plot.histogram_plt(
            ax,
            geo=rume.scope.select.by_state(state),
            time=rume.time_frame.select.days(0, 0),
            quantity=filter_output.select_parameter.by_name("test_0"),
            label_format="{q} @ {n}: {t}",
            hist_kwargs=[{"bins": 2}],
        )
        assert len(plots) == 1
        assert isinstance(plots[0], BarContainer)
        assert ax.get_legend_handles_labels()[1] == [
            f"*::*::test_0 @ {expected}: 2021-01-01"
        ]
        assert sum(bar.get_height() for bar in plots[0]) == 3


def test_histogram_plt_mixed_histogram_types(rume, filter_output):
    _, ax = plt.subplots()
    plots = filter_output.plot.histogram_plt(
        ax,
        geo=rume.scope.select.all(),
        time=rume.time_frame.select.days(0, 0),
        quantity=rume.ipm.select.compartments("S", "I"),
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


def test_histogram_plt_geography_and_time_aggregations(rume, filter_output):
    filter_output.compartments[..., 1] = 3
    _, ax = plt.subplots()
    plots = filter_output.plot.histogram_plt(
        ax,
        geo=rume.scope.select.all().sum(),
        time=rume.time_frame.select.all().agg(),
        quantity=rume.ipm.select.compartments("S", "I"),
        label_format="{n}: {q}: {t}",
        hist_kwargs=[],
    )
    assert len(plots) == 2
    assert ax.get_legend_handles_labels()[1] == [
        "*: S: 2021-01-01/2021-01-04",
        "*: I: 2021-01-01/2021-01-04",
    ]
    for plot, value in zip(plots, (2, 6), strict=True):
        assert isinstance(plot, BarContainer)
        assert sum(bar.get_height() for bar in plot) == 3
        bar = next(bar for bar in plot if bar.get_height() > 0)
        assert bar.get_x() <= value <= bar.get_x() + bar.get_width()


def test_histogram_plt_grouped_quantities(multistrata_filter_output):
    output = multistrata_filter_output
    rume = output.rume
    for name, value in (("S_young", 1), ("S_old", 4), ("I_young", 2), ("I_old", 8)):
        index = rume.ipm.select.compartments(name).compartment_index
        output.compartments[..., index] = value

    _, ax = plt.subplots()
    plots = output.plot.histogram_plt(
        ax,
        geo=rume.scope.select.by_state("04"),
        time=rume.time_frame.select.days(0, 0),
        quantity=rume.ipm.select.compartments("S_*", "I_*").group(strata=True).sum(),
        hist_kwargs=[{"bins": 1}],
    )
    assert ax.get_legend_handles_labels()[1] == [
        "AZ: S: 2021-01-01",
        "AZ: I: 2021-01-01",
    ]
    assert len(plots) == 2
    for plot, value in zip(plots, (5, 10), strict=True):
        assert isinstance(plot, BarContainer)
        assert len(plot) == 1
        assert plot[0].get_height() == 3
        assert plot[0].get_x() + plot[0].get_width() / 2 == pytest.approx(value)


def test_histogram_plt_rejects_multiple_times(rume, filter_output):
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match="single point"):
        filter_output.plot.histogram_plt(
            ax,
            geo=rume.scope.select.all(),
            time=rume.time_frame.select.all(),
            quantity=rume.ipm.select.compartments("S"),
        )


@pytest.mark.parametrize("legend", ["auto", "on", "off", "outside"])
def test_histogram_legend(rume, filter_output, legend):
    filter_output.plot.histogram(
        geo=rume.scope.select.all(),
        time=rume.time_frame.select.days(0, 0),
        quantity=rume.ipm.select.compartments("S", "I"),
        legend=legend,
    )
    assert (plt.gca().get_legend() is not None) is (legend != "off")


def test_histogram_plot_labels(rume, filter_output):
    filter_output.plot.histogram(
        geo=rume.scope.select.by_state("04"),
        time=rume.time_frame.select.days(0, 0),
        quantity=rume.ipm.select.compartments("S"),
        title="Forecast",
    )
    ax = plt.gca()
    assert ax.get_title() == "Forecast"
    assert ax.get_xlabel() == "value"
    assert ax.get_ylabel() == "count"


#############
# KDE plots #
#############


def test_kde_plt_transform_columns(rume, filter_output):
    for realization in range(3):
        filter_output.compartments[realization, ...] = realization
    seen = []

    def transform(data):
        seen.append(list(data.columns))
        return data

    _, ax = plt.subplots()
    filter_output.plot.kernel_density_plt(
        ax,
        geo=rume.scope.select.by_state("04"),
        time=rume.time_frame.select.days(0, 0),
        quantity=rume.ipm.select.compartments("S"),
        transform=transform,
    )
    assert seen == [["geo", "quantity", "value"]]


@pytest.mark.parametrize("bandwidth", ["silverman", 0.5])
def test_kde_plt_transform_values(rume, filter_output, bandwidth):
    for realization in range(3):
        filter_output.compartments[realization, ...] = realization
    _, ax = plt.subplots()
    lines = filter_output.plot.kernel_density_plt(
        ax,
        geo=rume.scope.select.by_state("04"),
        time=rume.time_frame.select.days(0, 0),
        quantity=rume.ipm.select.compartments("S"),
        bandwidth=bandwidth,
        delta_t=0.5,
        transform=lambda data: data.assign(value=data["value"] + 0.5),
    )
    expected_x = [0.5, 1, 1.5, 2, 2.5]
    np.testing.assert_allclose(lines[0].get_xdata(), expected_x)
    np.testing.assert_allclose(
        lines[0].get_ydata(),
        gaussian_kde([0.5, 1.5, 2.5], bw_method=bandwidth)(expected_x),
    )


@pytest.mark.parametrize(
    ("delta_t", "expected_x"),
    [(None, np.linspace(0, 2, 101)), (0.5, [0, 0.5, 1, 1.5, 2])],
)
def test_kde_plt_sampling_interval(rume, filter_output, delta_t, expected_x):
    for realization in range(3):
        filter_output.compartments[realization, ...] = realization
    _, ax = plt.subplots()
    lines = filter_output.plot.kernel_density_plt(
        ax,
        geo=rume.scope.select.by_state("04"),
        time=rume.time_frame.select.days(0, 0),
        quantity=rume.ipm.select.compartments("S"),
        delta_t=delta_t,
    )
    np.testing.assert_allclose(lines[0].get_xdata(), expected_x)


def test_kde_plt_subplots_and_parameters(rume, filter_output):
    values = next(iter(filter_output.estimated_params.values()))
    values[:, :, :] = np.arange(3)[:, np.newaxis, np.newaxis]
    _, axes = plt.subplots(1, 2)
    for ax, state, expected in zip(axes, ("04", "35"), ("AZ", "NM")):
        lines = filter_output.plot.kernel_density_plt(
            ax,
            geo=rume.scope.select.by_state(state),
            time=rume.time_frame.select.days(0, 0),
            quantity=filter_output.select_parameter.by_name("test_0"),
            label_format="{q} @ {n}: {t}",
        )
        assert len(lines) == 1
        assert lines[0].get_label() == (f"*::*::test_0 @ {expected}: 2021-01-01")


def test_kde_plt_geography_and_time_aggregations(rume, filter_output):
    for realization in range(3):
        filter_output.compartments[realization, ...] = realization
        filter_output.compartments[realization, ..., 1] = 3 * realization
    _, ax = plt.subplots()
    lines = filter_output.plot.kernel_density_plt(
        ax,
        geo=rume.scope.select.all().sum(),
        time=rume.time_frame.select.all().agg(),
        quantity=rume.ipm.select.compartments("S", "I"),
        delta_t=2,
    )
    assert [line.get_label() for line in lines] == [
        "*: S: 2021-01-01/2021-01-04",
        "*: I: 2021-01-01/2021-01-04",
    ]
    for line, values in zip(lines, ([0, 2, 4], [0, 6, 12]), strict=True):
        expected_x = np.arange(0, values[-1] + 1, 2)
        np.testing.assert_array_equal(line.get_xdata(), expected_x)
        np.testing.assert_allclose(line.get_ydata(), gaussian_kde(values)(expected_x))


def test_kde_plt_grouped_quantities(multistrata_filter_output):
    output = multistrata_filter_output
    rume = output.rume
    for name, value, step in (
        ("S_young", 1, 1),
        ("S_old", 4, 2),
        ("I_young", 2, 2),
        ("I_old", 8, 4),
    ):
        index = rume.ipm.select.compartments(name).compartment_index
        output.compartments[..., index] = (
            value + step * np.arange(output.num_realizations)[:, np.newaxis, np.newaxis]
        )

    _, ax = plt.subplots()
    lines = output.plot.kernel_density_plt(
        ax,
        geo=rume.scope.select.by_state("04"),
        time=rume.time_frame.select.days(0, 0),
        quantity=rume.ipm.select.compartments("S_*", "I_*").group(strata=True).sum(),
        delta_t=3,
    )
    assert [line.get_label() for line in lines] == [
        "AZ: S: 2021-01-01",
        "AZ: I: 2021-01-01",
    ]
    for line, values in zip(lines, ([5, 8, 11], [10, 16, 22]), strict=True):
        expected_x = np.arange(values[0], values[-1] + 1, 3)
        np.testing.assert_array_equal(line.get_xdata(), expected_x)
        np.testing.assert_allclose(line.get_ydata(), gaussian_kde(values)(expected_x))


def test_kde_plt_rejects_constant_data(rume, filter_output):
    _, ax = plt.subplots()
    with pytest.raises(np.linalg.LinAlgError, match="singular"):
        filter_output.plot.kernel_density_plt(
            ax,
            geo=rume.scope.select.by_state("04"),
            time=rume.time_frame.select.days(0, 0),
            quantity=rume.ipm.select.compartments("S"),
        )


def test_kde_plt_rejects_multiple_times(rume, filter_output):
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match="single point"):
        filter_output.plot.kernel_density_plt(
            ax,
            geo=rume.scope.select.all(),
            time=rume.time_frame.select.all(),
            quantity=rume.ipm.select.compartments("S"),
        )


@pytest.mark.parametrize("delta_t", [0, -0.5, float("nan"), float("inf")])
def test_kde_plt_rejects_invalid_delta_t(rume, filter_output, delta_t):
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match="delta_t"):
        filter_output.plot.kernel_density_plt(
            ax,
            geo=rume.scope.select.all(),
            time=rume.time_frame.select.days(0, 0),
            quantity=rume.ipm.select.compartments("S"),
            delta_t=delta_t,
        )


@pytest.mark.parametrize("legend", ["auto", "on", "off", "outside"])
def test_kde_legend(rume, filter_output, legend):
    for realization in range(3):
        filter_output.compartments[realization, ...] = realization
    filter_output.plot.kernel_density(
        geo=rume.scope.select.all(),
        time=rume.time_frame.select.days(0, 0),
        quantity=rume.ipm.select.compartments("S", "I"),
        legend=legend,
    )
    assert (plt.gca().get_legend() is not None) is (legend != "off")


def test_kde_plot_labels(rume, filter_output):
    for realization in range(3):
        filter_output.compartments[realization, ...] = realization
    filter_output.plot.kernel_density(
        geo=rume.scope.select.by_state("04"),
        time=rume.time_frame.select.days(0, 0),
        quantity=rume.ipm.select.compartments("S"),
        title="Forecast",
    )
    ax = plt.gca()
    assert ax.get_title() == "Forecast"
    assert ax.get_xlabel() == "value"
    assert ax.get_ylabel() == "density"
