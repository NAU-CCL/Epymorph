"""AdrioMaker library."""
from epymorph.geo.adrio.census.adrio_census import ADRIOMakerCensus
from epymorph.geo.adrio.file.adrio_file import ADRIOMakerCSV
from epymorph.geo.dynamic import ADRIOMaker

adrio_maker_library: dict[str, type[ADRIOMaker]] = {
    'Census': ADRIOMakerCensus,
    'CSV': ADRIOMakerCSV
}
