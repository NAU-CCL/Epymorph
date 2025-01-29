import numpy as np
from pandas import DataFrame

from epymorph.adrio import csv
from epymorph.geography.us_census import StateScope
from epymorph.geography.us_tiger import get_states


def test_csv(tmp_path):
    tmp_file = tmp_path / "population.csv"

    scope = StateScope.in_states(
        ["AZ", "FL", "GA", "MD", "NY", "NC", "SC", "VA"],
        year=2015,
    )
    to_postal_code = get_states(2015).state_fips_to_code

    # the values here are arbitrary, but should align with scope
    population = np.array(
        [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000], dtype=np.int64
    )

    # write csv file
    data_df = DataFrame(
        {
            "label": [to_postal_code[x] for x in scope.node_ids],
            "population": population,
        }
    ).sample(
        frac=1,
        random_state=np.random.RandomState(42),
    )  # put the rows in a (seeded) random order
    print(data_df)
    data_df.to_csv(tmp_file, header=False, index=False)

    # load the data
    actual = (
        csv.CSV(
            file_path=tmp_file,
            key_col=0,
            data_col=1,
            data_type=np.int64,
            key_type="state_abbrev",
            skiprows=None,
        )
        .with_context(
            # NOTE: to test filtering, load back a geographic subset of the initial data
            # this scope is the same as above but minus AZ and NY
            scope=StateScope.in_states(["12", "13", "24", "37", "45", "51"], year=2015),
        )
        .evaluate()
    )

    # compare with expected values
    expected = population[[1, 2, 3, 5, 6, 7]]

    assert np.array_equal(actual, expected)


# TODO: convert the rest of the cases in 2024-06-12
