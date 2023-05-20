import numpy as np

from epymorph.geo import Geo, validate_shape


def load() -> Geo:
    """
    US Counties/County-equivalents circa 2015:
    - geoids
    - labels
    - population
    - commuters
    - distance
    """

    with np.load(f"./epymorph/data/geo/us_counties_2015_geo.npz") as npz_data:
        data = dict(npz_data)

    n = len(data['labels'])
    validate_shape('labels', data['labels'], (n,))
    validate_shape('geoids', data['geoids'], (n,))
    validate_shape('population', data['population'], (n,))
    validate_shape('commuters', data['commuters'], (n, n))
    validate_shape('distance', data['distance'], (n, n))

    # Precompute data views:
    # Average commuters between node pairs.
    commuters = data['commuters']
    commuters_average = np.zeros(commuters.shape, dtype=np.double)
    for i in range(commuters.shape[0]):
        for j in range(i + 1, commuters.shape[1]):
            nbar = (commuters[i, j] + commuters[j, i]) // 2
            commuters_average[i, j] = nbar
            commuters_average[j, i] = nbar
    # Total commuters living in each node.
    commuters_by_node = commuters.sum(axis=1, dtype=np.int_)
    # Commuters as a ratio to the total commuters living in that node.
    commuting_probability = commuters / commuters_by_node[:, None]

    validate_shape('commuters_average', commuters_average, (n, n))
    validate_shape('commuters_by_node', commuters_by_node, (n,))
    validate_shape('commuting_probability', commuting_probability, (n, n))

    precomputed = {
        'commuters_average': commuters_average,
        'commuters_by_node': commuters_by_node,
        'commuting_probability': commuting_probability
    }

    return Geo(
        nodes=n,
        labels=data['labels'],
        data=data | precomputed
    )
