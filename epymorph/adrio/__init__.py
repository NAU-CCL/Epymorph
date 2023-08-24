from epymorph.adrio.adrio import ADRIO
from epymorph.adrio.census.centroid import Centroid
from epymorph.adrio.census.dissimilarity_index import DissimilarityIndex
from epymorph.adrio.census.geoid import GEOID
from epymorph.adrio.census.median_income import MedianIncome
from epymorph.adrio.census.name_and_state import NameAndState
from epymorph.adrio.census.pop_density_km2 import PopDensityKm2
from epymorph.adrio.census.population import Population
from epymorph.adrio.census.population_by_age import PopulationByAge

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
