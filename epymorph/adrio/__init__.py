from epymorph.adrio.adrio import ADRIOMaker
from epymorph.adrio.census.adrio_census import ADRIOMakerCensus

adrio_maker_library: dict[str, type[ADRIOMaker]] = {
    'Census': ADRIOMakerCensus
}
