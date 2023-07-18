from epymorph.adrio.adrio import ADRIO
from epymorph.adrio.uscounties.centroid import Centroid
from epymorph.adrio.uscounties.dissimilarity_index import DissimilarityIndex
from epymorph.adrio.uscounties.geoid import GEOID
from epymorph.adrio.uscounties.median_income import MedianIncome
from epymorph.adrio.uscounties.name_and_state import NameAndState
from epymorph.adrio.uscounties.pop_density_km2 import PopDensityKm2
from epymorph.adrio.uscounties.population import Population
from epymorph.adrio.uscounties.population_by_age import PopulationByAge

uscounties_library: dict[str, type[ADRIO]] = {
    'NameAndState': NameAndState,
    'GEOID': GEOID,
    'Centroid': Centroid,
    'MedianIncome': MedianIncome,
    'Population': Population,
    'PopulationByAge': PopulationByAge,
    'PopDensityKm2': PopDensityKm2,
    'DissimilarityIndex': DissimilarityIndex
}
