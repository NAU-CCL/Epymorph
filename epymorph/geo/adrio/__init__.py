from epymorph.geo.adrio.adrio import ADRIOMaker
from epymorph.geo.adrio.census.adrio_census import ADRIOMakerCensus

adrio_maker_library: dict[str, type[ADRIOMaker]] = {
    'Census': ADRIOMakerCensus
}
