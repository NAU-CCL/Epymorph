from epymorph.adrio.uscounties.dissimilarity_index import DissimilarityIndex
from epymorph.adrio.uscounties.geographic_centroid import GeographicCentroid
from epymorph.adrio.uscounties.median_income import MedianIncome
from epymorph.adrio.uscounties.name_state import NameState
from epymorph.adrio.uscounties.pop_by_age import PopByAge
from epymorph.adrio.uscounties.population_density import PopulationDensity

uscounties_library = {
    'NameState': NameState,
    'GeographicCentroid': GeographicCentroid,
    'MedianIncome': MedianIncome,
    'PopByAge': PopByAge,
    'PopulationDensity': PopulationDensity,
    'DissimilarityIndex': DissimilarityIndex
}
