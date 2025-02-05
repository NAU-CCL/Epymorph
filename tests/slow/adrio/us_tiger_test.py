from itertools import pairwise

import numpy as np

from epymorph.adrio import us_tiger
from epymorph.data_type import CentroidDType
from epymorph.geography.us_census import CountyScope
from epymorph.time import TimeFrame
from epymorph.util import match


def test_geometric_centroids_values():
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

    actual = (
        us_tiger.GeometricCentroid()
        .with_context(
            scope=CountyScope.in_counties(
                ["04001", "04003", "04005", "04013", "04017"],
                year=2020,
            ),
            time_frame=TimeFrame.year(2020),
        )
        .evaluate()
    )

    assert match.dtype(CentroidDType)(actual.dtype)
    assert np.allclose(
        expected["latitude"],
        actual["latitude"],
    )
    assert np.allclose(
        expected["longitude"],
        actual["longitude"],
    )


def calculate_expected_values():
    """
    The expected values for centroid were calculated using this function.
    Execute this file directly (`uv run python <path_to>/us_tiger_test.py`) to evaluate.
    """
    from geopandas import read_file

    scope = CountyScope.in_counties(
        ["04001", "04003", "04005", "04013", "04017"],
        year=2020,
    )

    # load in shapefile data for use in centroid caclulations
    gdf = read_file(
        "https://www2.census.gov/geo/tiger/TIGER2020/COUNTY/tl_2020_us_county.zip",
        engine="fiona",
        ignore_geometry=False,
        include_fields=["GEOID", "STUSPS"],
    )
    gdf = gdf[gdf["GEOID"].isin(scope.node_ids)]
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

    print(gdf["geometry"].apply(centroid).to_list())


if __name__ == "__main__":
    calculate_expected_values()
