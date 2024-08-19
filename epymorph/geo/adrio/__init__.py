"""AdrioMaker library."""
from epymorph.geo.adrio.census.adrio_census import ADRIOMakerCensus
from epymorph.geo.adrio.census.adrio_lodes import ADRIOMakerLODES
from epymorph.geo.adrio.disease.adrio_cdc import ADRIOMakerCDC
from epymorph.geo.adrio.file.adrio_csv import ADRIOMakerCSV
from epymorph.geo.dynamic import ADRIOMaker

adrio_maker_library: dict[str, type[ADRIOMaker]] = {
    'Census': ADRIOMakerCensus,
    'CSV': ADRIOMakerCSV,
    'LODES': ADRIOMakerLODES,
    'CDC': ADRIOMakerCDC
}
