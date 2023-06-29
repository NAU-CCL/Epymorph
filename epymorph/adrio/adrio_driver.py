from epymorph.adrio import uscounties_library
from epymorph.adrio.adrio import ADRIOSpec, GEOSpec, deserialize, serialize
from epymorph.geo import Geo
from epymorph.util import DataDict


def create_geo(file_path: str) -> Geo:
    # create GEOSpec object from file
    spec = deserialize(file_path)
    data = DataDict()
    current_data = []
    # loop for all ADRIOSpecs
    for i in range(len(spec.adrios)):
        # get adrio class from library dictionary (library hardcoded for now)
        current = uscounties_library.get(spec.adrios[i].class_name)
        # fetch data from adrio
        if current is not None:
            current_obj = current()
            current_data = current_obj.fetch(nodes=spec.nodes)
            data[current_obj.attribute] = current_data

    # build and return Geo (what to do for nodes/label?)
    return Geo(
        nodes=len(current_data),
        labels=[name[0] + ', ' + name[1] for name in data['name and state']],
        data=data
    )


# function to simplify GEOSpec creation (likely placeholder)
def create_geo_spec(id: str, nodes: list[str], adrios: list[str]):
    # create ADRIOSpecs
    ad_specs = list()
    for i in adrios:
        ad_specs.append(ADRIOSpec(i))

    spec = GEOSpec(id, nodes, ad_specs)

    # serialize GEOSpec
    serialize(spec, f'{id}.jsonpickle')
