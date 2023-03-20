from __future__ import annotations

import numpy as np

from geo import Geo, GeoParamN, GeoParamNN, GeoParamNT


def load_geo() -> Geo:
    """
    Pei-style geo model. Defines nodes representing 6 east-coast US states, each with:
    - total population
    - state-to-state commuters
    - absolute humidity by day (for 365 days)
    """
    pop_labels = ["FL", "GA", "MD", "NC", "SC", "VA"]
    humidity = np.loadtxt('./data/pei-humidity.csv',
                          delimiter=',', dtype=np.double)
    population = np.loadtxt('./data/pei-population.csv',
                            delimiter=',', dtype=np.int_)
    commuters = np.loadtxt('./data/pei-commuters.csv',
                           delimiter=',', dtype=np.int_)
    return Geo(pop_labels, [
        GeoParamN("population", population),
        GeoParamNT("humidity", humidity),
        GeoParamNN("commuters", commuters)
    ])
