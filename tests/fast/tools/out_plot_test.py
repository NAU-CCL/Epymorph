import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# See conftest for RUME and Output fixtures


def test_line_plt_ordering_and_transform(rume, output):
    fig, ax = plt.subplots()
    try:
        lines = output.plot.line_plt(
            ax,
            rume.scope.select.all(),
            rume.time_frame.select.all(),
            rume.ipm.select.compartments("S", "I"),
            ordering="quantity",
            line_kwargs=[{"color": "red"}, {"color": "blue"}],
            transform=lambda df: df.assign(value=df["value"] / 100),
        )

        assert [line.get_label() for line in lines] == [
            "AZ: S",
            "NM: S",
            "AZ: I",
            "NM: I",
        ]
        assert [line.get_color() for line in lines] == ["red", "blue", "red", "blue"]
        np.testing.assert_array_equal(lines[0].get_xdata(), [1, 2, 3, 4])
        np.testing.assert_array_equal(lines[0].get_ydata(), [30, 32, 34, 36])
        np.testing.assert_array_equal(lines[3].get_ydata(), [21, 23, 25, 27])
    finally:
        plt.close(fig)


def test_line_date_axis(rume, output):
    geo = rume.scope.select.by_state("04")
    time = rume.time_frame.select.all()
    quantity = rume.ipm.select.compartments("S")
    fig, ax = plt.subplots()
    try:
        line = output.plot.line_plt(ax, geo, time, quantity, time_format="date")[0]
        pd.testing.assert_index_equal(
            pd.DatetimeIndex(line.get_xdata()),
            pd.date_range("2021-01-02", periods=4),
        )
    finally:
        plt.close(fig)


def test_save_line_plot(rume, output, tmp_path):
    geo = rume.scope.select.by_state("04")
    time = rume.time_frame.select.all()
    quantity = rume.ipm.select.compartments("S")

    path = tmp_path / "plot.png"
    try:
        output.plot.line(geo, time, quantity, to_file=path)
        assert path.is_file()
        assert path.stat().st_size > 0
    finally:
        plt.close()
