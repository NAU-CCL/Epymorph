{"py/object": "epymorph.geo.spec.DynamicGeoSpec", "py/state": {"attributes": [{"py/object": "epymorph.simulation.AttributeDef", "py/state": ["label", "geo", {"py/type": "builtins.str"}, {"py/object": "epymorph.data_shape.Node"}, {"py/reduce": ["py/newobj", {"py/tuple": [{"py/type": "sympy.core.symbol.Symbol"}, "label"]}]}, null, null]}, {"py/object": "epymorph.simulation.AttributeDef", "py/state": ["population", "geo", {"py/type": "builtins.int"}, {"py/id": 5}, {"py/reduce": ["py/newobj", {"py/tuple": [{"py/type": "sympy.core.symbol.Symbol"}, "population"]}]}, null, null]}, {"py/object": "epymorph.simulation.AttributeDef", "py/state": ["population_by_age", "geo", {"py/type": "builtins.int"}, {"py/object": "epymorph.data_shape.NodeAndArbitrary", "index": 3}, {"py/reduce": ["py/newobj", {"py/tuple": [{"py/type": "sympy.core.symbol.Symbol"}, "population_by_age"]}]}, null, null]}, {"py/object": "epymorph.simulation.AttributeDef", "py/state": ["centroid", "geo", [{"py/tuple": ["longitude", {"py/type": "builtins.float"}]}, {"py/tuple": ["latitude", {"py/type": "builtins.float"}]}], {"py/id": 5}, {"py/reduce": ["py/newobj", {"py/tuple": [{"py/type": "sympy.core.symbol.Symbol"}, "centroid"]}]}, null, null]}, {"py/object": "epymorph.simulation.AttributeDef", "py/state": ["geoid", "geo", {"py/type": "builtins.str"}, {"py/id": 5}, {"py/reduce": ["py/newobj", {"py/tuple": [{"py/type": "sympy.core.symbol.Symbol"}, "geoid"]}]}, null, null]}, {"py/object": "epymorph.simulation.AttributeDef", "py/state": ["dissimilarity_index", "geo", {"py/type": "builtins.float"}, {"py/id": 5}, {"py/reduce": ["py/newobj", {"py/tuple": [{"py/type": "sympy.core.symbol.Symbol"}, "dissimilarity_index"]}]}, null, null]}, {"py/object": "epymorph.simulation.AttributeDef", "py/state": ["median_income", "geo", {"py/type": "builtins.int"}, {"py/id": 5}, {"py/reduce": ["py/newobj", {"py/tuple": [{"py/type": "sympy.core.symbol.Symbol"}, "median_income"]}]}, null, null]}, {"py/object": "epymorph.simulation.AttributeDef", "py/state": ["pop_density_km2", "geo", {"py/type": "builtins.float"}, {"py/id": 5}, {"py/reduce": ["py/newobj", {"py/tuple": [{"py/type": "sympy.core.symbol.Symbol"}, "pop_density_km2"]}]}, null, null]}, {"py/object": "epymorph.simulation.AttributeDef", "py/state": ["commuters", "geo", {"py/type": "builtins.int"}, {"py/object": "epymorph.data_shape.NodeAndNode"}, {"py/reduce": ["py/newobj", {"py/tuple": [{"py/type": "sympy.core.symbol.Symbol"}, "commuters"]}]}, null, null]}], "time_period": {"py/object": "epymorph.geo.spec.Year", "year": 2015}, "geography": {"py/object": "epymorph.geo.adrio.census.adrio_census.CensusGeography", "granularity": {"py/reduce": [{"py/type": "epymorph.geo.adrio.census.adrio_census.Granularity"}, {"py/tuple": [1]}]}, "filter": {"state": ["04", "08", "49", "35", "32"], "county": ["*"], "tract": ["*"], "block group": ["*"]}}, "source": {"label": "Census:name", "population": "Census", "population_by_age": "Census", "centroid": "Census", "geoid": "Census", "dissimilarity_index": "Census", "median_income": "Census", "pop_density_km2": "Census", "commuters": "Census"}}}