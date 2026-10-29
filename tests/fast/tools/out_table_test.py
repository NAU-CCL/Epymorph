import pandas as pd
import pytest

# See conftest for RUME and Output fixtures


def test_quantiles(rume, output):
    actual = output.table.quantiles(
        [0.0, 0.5, 1.0],
        rume.scope.select.by_state("04"),
        rume.time_frame.select.all(),
        rume.ipm.select.compartments("S", "I"),
    )
    expected = pd.DataFrame(
        {
            "geo": ["AZ", "AZ"],
            "quantity": ["S", "I"],
            "0.0": [3000.0, 2000.0],
            "0.5": [3300.0, 2300.0],
            "1.0": [3600.0, 2600.0],
        }
    )
    pd.testing.assert_frame_equal(actual, expected)

    actual_string = output.table.quantiles(
        [0.0, 0.5, 1.0],
        rume.scope.select.by_state("04"),
        rume.time_frame.select.all(),
        rume.ipm.select.compartments("S", "I"),
        result_format="string",
    )
    assert actual_string == expected.to_string()


def test_sum_excludes_compartments(rume, output):
    with pytest.warns(UserWarning, match="sum\\(\\).*removed"):
        actual = output.table.sum(
            rume.scope.select.by_state("04"),
            rume.time_frame.select.all(),
            rume.ipm.select.all(),
        )
    expected = pd.DataFrame(
        {
            "geo": ["AZ", "AZ", "AZ"],
            "quantity": ["S → I", "I → R", "R → S"],
            "sum": [1320, 920, 520],
        }
    )
    pd.testing.assert_frame_equal(actual, expected)
