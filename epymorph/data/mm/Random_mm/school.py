# Modeled after the Pei influenza paper, this model simulates
# two types of movers -- regular commuters and more-randomized dispersers.
# Each is somewhat stochastic but adhere to the general shape dictated
# by the commuters array. Both kinds of movement can be "tuned" through
# their respective parameters: move_control and theta.

[move-steps: per-day=3; duration=[1/3, 1/6, 1/2]]

[attrib: source=geo; name=school_commuters; shape=NxN; dtype=int;
    description="A node-to-node commuters matrix."]

[attrib: source=params; name=move_control; shape=S; dtype=float;
    description="A factor which modulates the number of commuters by conducting a binomial draw with this probability and the expected commuters from the commuters matrix."]

[attrib: source=params; name=theta; shape=S; dtype=float;
    description="A factor which allows for randomized movement by conducting a poisson draw with this factor times the average number of commuters between two nodes from the commuters matrix."]


[attrib: source=geo; name=population; shape=N; dtype=int;
    description="The total population at each node."]

[attrib: source=geo; name=centroid; shape=N; dtype=[(longitude, float), (latitude, float)];
    description="The centroids for each node as (longitude, latitude) tuples."]

[predef: function = 
def pei_movement():
    """Pei style movement pre definition"""
    commuters = geo['school_commuters']
    
    commuters_average = (commuters + commuters.T) // 2
    
    commuters_by_node = np.sum(commuters, axis=1)
    
    commuting_probability = row_normalize(commuters)

    centroid = geo['centroid']
    distance = pairwise_haversine(centroid['longitude'], centroid['latitude'])
    dispersal_kernel = row_normalize(1 / np.exp(distance / 15))
    weekend_kernel = row_normalize(1 / np.exp(distance / 30))

    return {
        'commuters_average': commuters_average, 
        'commuters_by_node': commuters_by_node,
        'commuting_probability': commuting_probability,
        'dispersal_kernel': dispersal_kernel
    }
]

# Commuter movement
[mtype: days=[M,T,W,Th,F]; leave=1; duration=0d; return=2; function=
def commuters(t):
    typical = predef['commuters_by_node']
    actual = np.binomial(typical, params['move_control'])
    return np.multinomial(actual, predef['commuting_probability'])
]

[mtype: days=[M,T,W,Th,F]; leave=2; duration=0d; return=3; function=
def centroids_commuters(t):
    n_commuters = np.floor(geo['population'] * 0.1).astype(SimDType)
    return np.multinomial(n_commuters, predef['dispersal_kernel'])
]

[mtype: days=[Sa,Su]; leave=1; duration=0d; return=3; function=
def centroids_commuters(t):
    n_commuters = np.floor(geo['population'] * 0.1).astype(SimDType)
    return np.multinomial(n_commuters, predef['dispersal_kernel'])
]

# Random dispersers movement
[mtype: days=all; leave=1; duration=0d; return=2; function=
def dispersers(t):
    avg = predef['commuters_average']
    return np.poisson(avg * params['theta'])
]
