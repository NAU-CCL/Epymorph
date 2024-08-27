# epymorph

Prototype EpiMoRPH system written in Python. It is usable as a code library, for instance from within a Jupyter Notebook.

See the `USAGE.ipynb` Notebook for a basic usage example.

See `CONTRIBUTING.md` for development environment setup instructions.

## Configuration

epymorph accepts configuration values provided by your system's environment variables. This may include settings which change the behavior of epymorph, or secrets like API keys needed to interface with third-party services. All values are optional unless you are using a feature which requires them.

Currently supported values include:

- `CENSUS_API_KEY`: your API key for the US Census API ([which you can request here](https://api.census.gov/data/key_signup.html))
- `EPYMORPH_CACHE_PATH`: the path epymorph should use to cache files; this defaults to a location appropriate to your operating system for cached files
