# epymorph

The `epymorph` package is the product of the EpiMoRPH (Epidemiological Modeling Resources for Public Health) project, and aims to provide a simplified framework for completing the full lifecycle of a spatial modeling experiment. epymorph streamlines methods for building, simulating, and fitting, metapopulation models of infectious pathogens. This Python package is easily accessible to beginning modelers, while also sophisticated enough to allow rapid design and execution of complex modeling experiments by highly experienced modelers. Specific aims include dramatic streamlining of model building speed, increased model transparency, automated fitting of models to observed data, and easy transportability of models across temporal and geographic scenarios.

Read the [documentation at docs.epimorph.org](https://docs.www.epimorph.org).

For general inquiries please contact us via email at Epymorph@nau.edu

See [CONTRIBUTING.md](CONTRIBUTING.md) for more information on how to contribute to the codebase.

## Configuration

epymorph accepts configuration values provided by your system's environment variables. This may include settings which change the behavior of epymorph, or secrets like API keys needed to interface with third-party services. All values are optional unless you are using a feature which requires them.

Currently supported values include:

- `CENSUS_API_KEY`: your API key for the US Census API ([which you can request here](https://api.census.gov/data/key_signup.html))
- `EPYMORPH_CACHE_PATH`: the path epymorph should use to cache files; this defaults to a location appropriate to your operating system for cached files
