from epymorph.adrio.adrio import ADRIO
from epymorph.adrio.census.average_household_size import AverageHouseholdSize
from epymorph.adrio.census.centroid import Centroid
from epymorph.adrio.census.commuters import Commuters
from epymorph.adrio.census.dissimilarity_index import DissimilarityIndex
from epymorph.adrio.census.geoid import GEOID
from epymorph.adrio.census.gini_index import GiniIndex
from epymorph.adrio.census.median_age import MedianAge
from epymorph.adrio.census.median_income import MedianIncome
from epymorph.adrio.census.name import Name
from epymorph.adrio.census.pop_density_km2 import PopDensityKm2
from epymorph.adrio.census.population import Population
from epymorph.adrio.census.population_by_age import PopulationByAge
from epymorph.adrio.census.population_by_age_x6 import PopulationByAgex6
from epymorph.adrio.census.tract_median_income import TractMedianIncome

uscounties_library: dict[str, type[ADRIO]] = {
    'Name': Name,
    'GEOID': GEOID,
    'Centroid': Centroid,
    'MedianIncome': MedianIncome,
    'Population': Population,
    'PopulationByAge': PopulationByAge,
    'PopDensityKm2': PopDensityKm2,
    'DissimilarityIndex': DissimilarityIndex,
    'TractMedianIncome': TractMedianIncome,
    'MedianAge': MedianAge,
    'AverageHouseholdSize': AverageHouseholdSize,
    'PopulationByAgex6': PopulationByAgex6,
    'GiniIndex': GiniIndex,
    'Commuters': Commuters
}
