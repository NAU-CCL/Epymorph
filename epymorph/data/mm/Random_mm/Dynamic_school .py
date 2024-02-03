from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from epymorph.data import registry
from epymorph.data_shape import Shapes
from epymorph.data_type import CentroidType, SimDType
from epymorph.movement_model import EveryDay, MovementClause, MovementModel
from epymorph.simulation import AttributeDef, Tick, TickDelta, TickIndex
from epymorph.util import pairwise_haversine, row_normalize


class DynamicClause(MovementClause):
    """The clause of the dynamic model."""
    requirements = (
        AttributeDef('population', int, Shapes.N,
                     comment="The total population at each node."),

        AttributeDef('centroid', CentroidType, Shapes.N,
                     comment="The centroids for each node as (longitude, latitude) tuples."),

        # Movement patterns for 1< km 
        AttributeDef('distance_0km', float, Shapes.S,
                                    comment="A timeseries of real movement data provide by META"),
        AttributeDef('distance_phi', float, Shapes.S, default_value=40.0,
                                    comment = 'Influences the distance that movers tend to travel'),

        # Movement patterns for 1 km to 10 km 
        AttributeDef('distance_0_10km', float, Shapes.S, default_value=40.0,
                                        comment="A node-to-node LODES 8 commuters matrix."),
        AttributeDef('short_distance_phi', float, Shapes.S,
                                    comment = 'Influences the distance that movers tend to travel'),

        # Movement patterns for 10 km to 100 km 
        AttributeDef('distance_10_100km', float, Shapes.S, default_value=40.0,
                                        comment="A node-to-node LODES 8 commuters matrix."),
        AttributeDef('medium_distance_phi', float, Shapes.S,
                                    comment = 'Influences the distance that movers tend to travel'),

        # Movement patterns for 100 km +
        AttributeDef('distance_100km', float, Shapes.S, default_value=40.0,
                                        comment="A node-to-node LODES 8 commuters matrix."),
        AttributeDef('long_distance_phi', float, Shapes.S,
                                    comment = 'Influences the distance that movers tend to travel'),
    )

    predicate = EveryDay()
    leaves = TickIndex(step=0)
    returns = TickDelta(step=1, days=0)

    @cached_property
    def dispersal_kernal_0km(self) -> NDArray(np.float64):
        """
        Utilizing real data provided by Data for Good 
        The goal of this dispersal kernal is simulate movement patterens of a given county within a state.
        This dispersal kernal simulate the movement patterns of individual who are considered to move with a 1km radius of their home location. 
        """
       
        centroid = self.data('centroid')
        phi = self.data('distance_phi')

        distance_kernel = np.zeros_like(distance) # Could this used as a cached propety
        distance = pairwise_haversine(centroid['longitude'], centroid['latitude']) # Could this used as a cached propety
        distance_indices = distance < 0.621371 # 1km = 0.621371 mi
        distance_kernel[distance_indices] = 1 / np.exp(distance[distance_indices] / phi)

        return row_normalize(distance_kernel)
    
    @cached_property
    def dispersal_kernal_0_10km(self) -> NDArray(np.float64):
        """
        Utilizing real data provided by Data for Good 
        The goal of this dispersal kernal is simulate movement patterens of a given county within a state.
        This dispersal kernal simulate the movement patterns of individual who are considered to move with a <1km to 10km radius of their home location. 
        """
        centroid = self.data('centroid')
        phi = self.data('short_distance_phi')
        
        distance_kernel = np.zeros_like(distance)
        distance = pairwise_haversine(centroid['longitude'], centroid['latitude'])
        short_distance_indices = (distance >= 0.621371) & (distance <= 6.21371) # 10km = 6.21371 mi
        distance_kernel[short_distance_indices] = 1 / np.exp(distance[short_distance_indices] / phi)

        return row_normalize(distance_kernel)

    @cached_property
    def dispersal_kernal_10_100km(self) -> NDArray(np.float64):
        """
        Utilizing real data provided by Data for Good 
        The goal of this dispersal kernal is simulate movement patterens of a given county within a state.
        This dispersal kernal simulate the movement patterns of individual who are considered to move with a 10km to 100km  of their home location. 
        """
        centroid = self.data('centroid')
        phi = self.data('medium_distance_phi')

        distance_kernel = np.zeros_like(distance)
        distance = pairwise_haversine(centroid['longitude'], centroid['latitude'])
        medium_distance_indices = (distance > 6.21371) & (distance < 62.1371) # 100km = 62.1371 mi
        distance_kernel[medium_distance_indices] = 1 / np.exp(distance[medium_distance_indices] / phi)

        return row_normalize(distance_kernel)

    @cached_property
    def dispersal_kernal_100km(self) -> NDArray(np.float64):
        """
        Utilizing real data provided by Data for Good 
        The goal of this dispersal kernal is simulate movement patterens of a given county within a state.
        This dispersal kernal simulate the movement patterns of individual who are considered to move with a 100km+ radius of their home location. 
        """
        centroid = self.data('centroid')
        phi = self.data('long_distance_phi')
        
        distance_kernel = np.zeros_like(distance)
        distance = pairwise_haversine(centroid['longitude'], centroid['latitude'])
        long_distance_indices = distance >= 62.1371 # 100+ km = 62.1371
        distance_kernel[long_distance_indices] = 1 / np.exp(distance[long_distance_indices] / phi)

        return row_normalize(distance_kernel)

    
    def evaluate(self, tick: Tick) -> NDArray[np.int64]:
        pop = self.data('population')

        n_commuters_0km = np.floor(pop * self.data('distance_0km')).astype(SimDType)
        n_commuters_0_10km = np.floor(pop * self.data('distance_0_10km')).astype(SimDType)
        n_commuters_10_100km = np.floor(pop * self.data('distance_10_100km')).astype(SimDType)
        n_commuters_100km = np.floor(pop * self.data('distance_100km')).astype(SimDType)

        mover_0km = self.rng.multinomial(n_commuters_0km, self.dispersal_kernal_0km) 
        mover_0_10km = self.rng.multinomial(n_commuters_0_10km, self.dispersal_kernal_0km) 
        mover_10_100km = self.rng.multinomial(n_commuters_10_100km, self.dispersal_kernal_0km) 
        mover_100km = self.rng.multinomial(n_commuters_100km, self.dispersal_kernal_0km) 

        return mover_0km, mover_0_10km, mover_10_100km, mover_100km

@registry.mm('dynamic')
class Dynamic(MovementModel):
    """
    The dynamic MM descibes the movement patterns of individual using celluar ping location 
    provide by Data for Good who's parent company is META. The goal of this movement model is
    to simulate movement patterns of individual on a day to day basis with provided information. 
    These individual travel to another location for 1/3 of the day, and the return home for the 
    remaining 2/3 of the day.

    """
    steps = (1 / 3, 2 / 3)
    clauses = (DynamicClause(),)
