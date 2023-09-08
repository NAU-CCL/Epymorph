import numpy as np

from epymorph.geo import CentroidDType, Geo


def load() -> Geo:
    label = ['AZ']
    return Geo(
        nodes=len(label),
        labels=label,
        data={
            'label': np.array(label, dtype=str),
            'geoid': np.array(['04'], dtype=str),
            'centroid': np.array([(-111.856111, 34.566667)], dtype=CentroidDType),
            'population': np.array([100_000], dtype=np.int64),
            'commuters': np.array([[0]], dtype=np.int64)
        }
    )
