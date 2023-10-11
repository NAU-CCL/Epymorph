{"py/object": "epymorph.geo.geo.GEOSpec", "id": "pei", "attributes": [{"py/object": "epymorph.geo.common.AttribDef", "py/newargs": {"py/tuple": ["label", {"py/type": "numpy.str_"}]}, "py/seq": ["label", {"py/type": "numpy.str_"}]}, {"py/object": "epymorph.geo.common.AttribDef", "py/newargs": {"py/tuple": ["population", {"py/type": "numpy.int64"}]}, "py/seq": ["population", {"py/type": "numpy.int64"}]}, {"py/object": "epymorph.geo.common.AttribDef", "py/newargs": {"py/tuple": ["geoid", {"py/type": "numpy.int64"}]}, "py/seq": ["geoid", {"py/type": "numpy.int64"}]}, {"py/object": "epymorph.geo.common.AttribDef", "py/newargs": {"py/tuple": ["centroid", {"py/reduce": [{"py/type": "numpy.dtype"}, {"py/tuple": ["V16", false, true]}, {"py/tuple": [3, "|", null, {"py/tuple": ["longitude", "latitude"]}, {"longitude": {"py/tuple": [{"py/reduce": [{"py/type": "numpy.dtype"}, {"py/tuple": ["f8", false, true]}, {"py/tuple": [3, "<", null, null, null, -1, -1, 0]}]}, 0]}, "latitude": {"py/tuple": [{"py/id": 8}, 8]}}, 16, 1, 16]}]}]}, "py/seq": ["centroid", {"py/id": 6}]}, {"py/object": "epymorph.geo.common.AttribDef", "py/newargs": {"py/tuple": ["commuters", {"py/type": "numpy.int64"}]}, "py/seq": ["commuters", {"py/type": "numpy.int64"}]}], "granularity": 0, "nodes": {"state": ["12", "13", "45", "37", "51", "24"], "county": ["*"], "tract": ["*"], "block group": ["*"]}, "year": 2015, "type": "Dynamic", "source": {"label": "Census:name", "population": "Census", "geoid": "Census", "centroid": "Census", "commuters": "Census"}}