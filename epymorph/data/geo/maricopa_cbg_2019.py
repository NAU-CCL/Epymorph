from importlib.resources import as_file, files

import numpy as np

from epymorph.geo import Geo, validate_shape

age_brackets = ["0-19", "20-34", "35-54", "55-64", "65-74", "75+"]


def load() -> Geo:
    """
    Maricopa County, AZ Census Block Groups circa 2019:
    - labels: CBG GEOID
    - population: total population, people (int)
    - centroid: CBG centroid, longitude and latitude (two floats)
    - median_age: median age, years (float)
    - pop_by_age: population by age, persons (six ints)
    - median_income: median income, past 12 months, infl. adj., dollars (int)
    - total_income: total income, past 12 months, infl. adj., dollars (int)
    - average_household_size: average household size, people (float)
    - pop_density_km2: population density, persons per km^2 (float)
    - tract_median_income: median income of the CBG's tract, past 12 months, infl. adj., dollars (int)
    - tract_gini_index: Gini index of the CBG's tract (float)
    """

    file = files('epymorph.data.geo').joinpath('maricopa_cbg_2019_geo.npz')
    with as_file(file) as f:
        npz_data = np.load(f)
        data = dict(npz_data)
        npz_data.close()

    n = len(data["labels"])
    validate_shape("labels", data["labels"], (n,), str)
    validate_shape("population", data["population"], (n,), int)
    validate_shape("centroid", data["centroid"], (n, 2), float)
    validate_shape("median_age", data["median_age"], (n,), float)
    validate_shape("pop_by_age", data["pop_by_age"], (n, 6), int)
    validate_shape("median_income", data["median_income"], (n,), int)
    validate_shape("total_income", data["total_income"], (n,), int)
    validate_shape(
        "average_household_size", data["average_household_size"], (n,), float
    )
    validate_shape("pop_density_km2", data["pop_density_km2"], (n,), float)
    validate_shape("tract_median_income", data["tract_median_income"], (n,), int)
    validate_shape("tract_gini_index", data["tract_gini_index"], (n,), float)

    return Geo(nodes=n, labels=data["labels"], data=data)
