[move-steps: per-day=3; duration=[1/3, 1/6, 1/2]]

#################### Dynamic Attribute ####################
[attrib: source=geo; name=population; shape=N; dtype=int;
    description="The total population at each node."]

[attrib: source=geo; name=centroid; shape=N; dtype=[(longitude, float), (latitude, float)];
    description="The centroids for each node as (longitude, latitude) tuples."]

#################### Data for Good Numpy Array ####################
[attrib: source=params; name= distance_0km; shape=S; dtype=float;
    description="Influences the distance that movers tend to travel."]

[attrib: source=params; name= distance_0_10km; shape=S; dtype=float;
    description="Influences the distance that movers tend to travel."]

[attrib: source=params; name= distance_10_100km; shape=S; dtype=float;
    description="Influences the distance that movers tend to travel."]

[attrib: source=params; name=distance_100km; shape=S; dtype=float;
    description="Influences the distance that movers tend to travel."]

#################### Phi values for Gavity Model ####################
[attrib: source=params; name= distance_phi; shape=S; dtype=float;
    description="Influences the distance that movers tend to travel."]

[attrib: source=params; name=short_distance_phi; shape=S; dtype=float;
    description="Influences the distance that movers tend to travel."]

[attrib: source=params; name=medium_distance_phi; shape=S; dtype=float;
    description="Influences the distance that movers tend to travel."]

[attrib: source=params; name=long_distance_phi; shape=S; dtype=float;
    description="Influences the distance that movers tend to travel."]

#################### School Attribute ####################
[attrib: source=geo; name=school_commuters; shape=NxN; dtype=int;
    description="A node-to-node commuters matrix."]

[attrib: source=params; name=move_control; shape=S; dtype=float;
    description="A factor which modulates the number of commuters by conducting a binomial draw with this probability and the expected commuters from the commuters matrix."]

[attrib: source=params; name=theta; shape=S; dtype=float;
    description="A factor which allows for randomized movement by conducting a poisson draw with this factor times the average number of commuters between two nodes from the commuters matrix."]

[predef: function = 
def movement():
    ############################## School Movement #############################
    commuters = geo['school_commuters']
    commuters_average = (commuters + commuters.T) // 2
    commuters_by_node = np.sum(commuters, axis=1)
    commuting_probability = row_normalize(commuters)

    ############################## Distance cutoffs #############################
    centroid = geo['centroid']
    distance = pairwise_haversine(centroid['longitude'], centroid['latitude'])

    distance_indices = distance < 0.621371 # 1km = 0.621371 mi
    short_distance_indices = (distance >= 0.621371) & (distance <= 6.21371) # 10km = 6.21371 mi
    medium_distance_indices = (distance > 6.21371) & (distance < 62.1371) # 100km = 62.1371 mi
    long_distance_indices = distance >= 62.1371 # 100+ km = 62.1371

    ############################### Dispersal Kernel ##############################
    distance_kernel = np.zeros_like(distance)
    distance_1km_10km_kernel = np.zeros_like(distance)
    distance_10km_100km_kernel = np.zeros_like(distance)
    distance_100km_kernel = np.zeros_like(distance)

    distance_kernel[distance_indices] = 1 / np.exp(distance[distance_indices] / params['distance_phi'])
    distance_1km_10km_kernel[short_distance_indices] = 1 / np.exp(distance[short_distance_indices] / params['short_distance_phi'])
    distance_10km_100km_kernel[medium_distance_indices] = 1 / np.exp(distance[medium_distance_indices] / params['medium_distance_phi'])
    distance_100km_kernel[long_distance_indices] = 1 / np.exp(distance[long_distance_indices] / params['long_distance_phi'])

    distance_kernel = row_normalize(distance_kernel)
    distance_1km_10km_kernel = row_normalize(distance_1km_10km_kernel)
    distance_10km_100km_kernel = row_normalize(distance_10km_100km_kernel)
    distance_100km_kernel = row_normalize(distance_100km_kernel)

    return {
        'commuters_average': commuters_average, 
        'commuters_by_node': commuters_by_node,
        'commuting_probability': commuting_probability,
        'distance_kernel': distance_kernel,
        'distance_1km_10km_kernel': distance_1km_10km_kernel,
        'distance_10km_100km_kernel': distance_10km_100km_kernel,
        'distance_100km_kernel': distance_100km_kernel,
    }
]

# Commuter movement
[mtype: days=[M,T,W,Th,F]; leave=1; duration=0d; return=2; function=
def commuters(t):
    typical = predef['commuters_by_node']
    actual = np.binomial(typical, params['move_control'])
    return np.multinomial(actual, predef['commuting_probability'])
]

# Random dispersers movement
[mtype: days=all; leave=1; duration=0d; return=2; function=
def dispersers(t):
    avg = predef['commuters_average']
    return np.poisson(avg * params['theta'])
]

# Commuter movement: assume 10% of the population are commuters
[mtype: days=[M,T,W,Th,F]; leave=2; duration=0d; return=3; function=
def dynamic_1km_movement(t):
    population = geo['population']
    ########################### Params Numpy Array #############################
    distance_0km = params['distance_0km'][t.day]

    #################### Fraction of the population moving ####################
    staying_at_home = np.floor(population * distance_0km)

    n_commuters = staying_at_home.astype(SimDType)
    return np.multinomial(n_commuters, predef['distance_kernel'])
]

[mtype: days=[M,T,W,Th,F]; leave=2; duration=0d; return=3; function=
def dynamic_1km_10km_movement(t):
    population = geo['population']
    ########################### Params Numpy Array #############################
    distance_0_10km = params['distance_0_10km'][t.day]

    #################### Fraction of the population moving ####################
    short_distance_movers = np.floor(population * distance_0_10km)

    n_commuters = short_distance_movers.astype(SimDType)
    return np.multinomial(n_commuters, predef['distance_1km_10km_kernel'])
]

[mtype: days=[M,T,W,Th,F]; leave=2; duration=0d; return=3; function=
def dynamic_10km_100km_movement(t):
    population = geo['population']
    ########################### Params Numpy Array #############################
    distance_10_100km = params['distance_10_100km'][t.day]

    #################### Fraction of the population moving ####################
    medium_distance_movers = np.floor(population * distance_10_100km)

    n_commuters = medium_distance_movers.astype(SimDType)
    return np.multinomial(n_commuters, predef['distance_10km_100km_kernel'])
]

[mtype: days=[M,T,W,Th,F]; leave=2; duration=0d; return=3; function=
def dynamic_100km_movement(t):
    population = geo['population']
    ########################### Params Numpy Array #############################
    distance_100km = params['distance_100km'][t.day]

    #################### Fraction of the population moving ####################
    long_distance_movers = np.floor(population * distance_100km)

    n_commuters = long_distance_movers.astype(SimDType)
    return np.multinomial(n_commuters, predef['distance_100km_kernel'])
]

########################### Weekend Movement #############################

# Commuter movement: assume 10% of the population are commuters
[mtype: days=[Sa,Su]; leave=1; duration=0d; return=3; function=
def dynamic_1km_movement_Weekend(t):
    population = geo['population']
    ########################### Params Numpy Array #############################
    distance_0km = params['distance_0km'][t.day]

    #################### Fraction of the population moving ####################
    staying_at_home = np.floor(population * distance_0km)

    n_commuters = staying_at_home.astype(SimDType)
    return np.multinomial(n_commuters, predef['distance_kernel'])
]

[mtype: days=[Sa,Su]; leave=1; duration=0d; return=3; function=
def dynamic_1km_10km_movement_Weekend(t):
    population = geo['population']
    ########################### Params Numpy Array #############################
    distance_0_10km = params['distance_0_10km'][t.day]

    #################### Fraction of the population moving ####################
    short_distance_movers = np.floor(population * distance_0_10km)

    n_commuters = short_distance_movers.astype(SimDType)
    return np.multinomial(n_commuters, predef['distance_1km_10km_kernel'])
]

[mtype: days=[Sa,Su]; leave=1; duration=0d; return=3; function=
def dynamic_10km_100km_movement_Weekend(t):
    population = geo['population']
    ########################### Params Numpy Array #############################
    distance_10_100km = params['distance_10_100km'][t.day]

    #################### Fraction of the population moving ####################
    medium_distance_movers = np.floor(population * distance_10_100km)

    n_commuters = medium_distance_movers.astype(SimDType)
    return np.multinomial(n_commuters, predef['distance_10km_100km_kernel'])
]

[mtype: days=[Sa,Su]; leave=1; duration=0d; return=3; function=
def dynamic_100km_movement_Weekend(t):
    population = geo['population']
    ########################### Params Numpy Array #############################
    distance_100km = params['distance_100km'][t.day]

    #################### Fraction of the population moving ####################
    long_distance_movers = np.floor(population * distance_100km)

    n_commuters = long_distance_movers.astype(SimDType)
    return np.multinomial(n_commuters, predef['distance_100km_kernel'])
]
