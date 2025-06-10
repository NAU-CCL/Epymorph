# ruff: noqa: T201
from itertools import pairwise

import numpy as np
import pytest

from epymorph.adrio import us_tiger
from epymorph.adrio.adrio import ADRIOContextError
from epymorph.data_type import CentroidDType
from epymorph.geography.us_census import CountyScope, StateScope
from epymorph.util import match


@pytest.fixture
def county_context():
    return {
        "scope": CountyScope.in_counties(
            ["04001", "04003", "04005", "04013", "04017"],
            year=2020,
        )
    }


def test_geometric_centroids_values(county_context):
    # values calculated manually using polygon centroid formula
    # applied to TIGRIS shapefile polygons
    # (see function `calculate_expected_values()` below)
    expected = np.array(
        [
            (-109.48884962248498, 35.39552879677974),
            (-109.75126313676874, 31.87963708630415),
            (-111.77052095609857, 35.838724829519194),
            (-112.49151143850366, 33.349039435609264),
            (-110.32141934757458, 35.39955033687498),
        ],
        dtype=CentroidDType,
    )

    actual = us_tiger.GeometricCentroid().with_context(**county_context).evaluate()

    assert match.dtype(CentroidDType)(actual.dtype)
    assert np.allclose(expected["latitude"], actual["latitude"])
    assert np.allclose(expected["longitude"], actual["longitude"])


def test_internal_points_values(county_context):
    # values from TIGER files
    # (see function `calculate_expected_values()` below)
    expected = np.array(
        [
            (-109.4901722, 35.385084),
            (-109.7751627, 31.8401287),
            (-111.7737277, 35.8296919),
            (-112.4989296, 33.3451756),
            (-110.3210248, 35.3907852),
        ],
        dtype=CentroidDType,
    )

    actual = us_tiger.InternalPoint().with_context(**county_context).evaluate()

    assert match.dtype(CentroidDType)(actual.dtype)
    assert np.allclose(expected["latitude"], actual["latitude"])
    assert np.allclose(expected["longitude"], actual["longitude"])


def test_names_values(county_context):
    # values from TIGER files
    # (see function `calculate_expected_values()` below)
    expected = np.array(
        ["Apache", "Cochise", "Coconino", "Maricopa", "Navajo"],
        dtype=np.str_,
    )

    actual = us_tiger.Name().with_context(**county_context).evaluate()

    assert np.array_equal(expected, actual)


def test_land_area_values(county_context):
    # values from TIGER files
    # (see function `calculate_expected_values()` below)
    expected = np.array(
        [29003486215, 16083178104, 48216135399, 23832530727, 25769061034],
        dtype=np.float64,
    )

    actual = us_tiger.LandAreaM2().with_context(**county_context).evaluate()

    assert np.array_equal(expected, actual)


def test_postal_code_counties(county_context):
    with pytest.raises(ADRIOContextError, match="StateScopes"):
        us_tiger.PostalCode().with_context(**county_context).evaluate()


def test_postal_code_values(county_context):
    expected = np.array(["AZ", "CO", "NM"], dtype=np.str_)
    actual = (
        us_tiger.PostalCode()
        .with_context(scope=StateScope.in_states(["04", "08", "35"], year=2020))
        .evaluate()
    )
    assert np.array_equal(expected, actual)


def calculate_expected_values():
    """
    The expected values for centroid were calculated using this function.
    Execute this file directly (`uv run python <path_to>/us_tiger_test.py`) to evaluate.
    """
    from geopandas import read_file

    node_ids = ["04001", "04003", "04005", "04013", "04017"]

    # load in shapefile data for use in centroid caclulations
    gdf = read_file(
        "https://www2.census.gov/geo/tiger/TIGER2020/COUNTY/tl_2020_us_county.zip",
        engine="fiona",
        ignore_geometry=False,
    )
    gdf = gdf[gdf["GEOID"].isin(node_ids)]
    gdf = gdf.sort_values(by="GEOID")

    # calculate centroids manually using polygon centroid formula:
    # https://en.wikipedia.org/wiki/Centroid#Of_a_polygon
    def centroid(polygon):
        shoelace, x_sum, y_sum = 0, 0, 0
        for (ax, ay), (bx, by) in pairwise(polygon.exterior.coords):
            s = ax * by - bx * ay
            shoelace += s
            x_sum += (ax + bx) * s
            y_sum += (ay + by) * s

        a = 0.5 * shoelace
        cx = x_sum / (6 * a)
        cy = y_sum / (6 * a)
        return (cx, cy)

    print("geometric centroids:")
    print(gdf["geometry"].apply(centroid).to_list())
    print()
    print("internal points:")
    print(list(zip(gdf["INTPTLON"].astype(float), gdf["INTPTLAT"].astype(float))))
    print()
    print("names:")
    print(gdf["NAME"].to_list())
    print()
    print("land area:")
    print((gdf["ALAND"]).to_list())


if __name__ == "__main__":
    calculate_expected_values()
