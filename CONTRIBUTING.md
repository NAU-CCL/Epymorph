# Contributing to epymorph

epymorph is open source software and we welcome your contributions and feedback.

For general inquiries please contact us via email at Epymorph@nau.edu

You can contribute directly to the codebase by [submitting Issues](https://github.com/NAU-CCL/Epymorph/issues) if you encounter bugs. We are also happy to consider [Pull Requests](https://github.com/NAU-CCL/Epymorph). In order to help us respond as fast as possible, please provide as much descriptive detail in your requests as you can. You might include code samples, error messages, input data, information about your operating system and Python version, and anything else that may be relevant.

## Development Setup

A correctly configured development environment is critical for contributing code to the project. For epymorph, that includes the following pieces:

- the epymorph git repository, cloned to your development machine,
- a python runtime of an appropriate version,
- a virtual environment with dependencies installed,
- an automated code formatter and linter, and
- your IDE of choice (integrated development environment).

While there are many options that fulfill the above, our "official" recommended environment includes the following:

- **uv** as a tool for python, project, and virtual environment management,
- **ruff** as a code formatter and linter, and
- **VS Code** as an IDE that integrates these tools.

To set up our recommended environment follow these steps:

1. [Install **uv**.](https://docs.astral.sh/uv/)
1. [Install **VS Code**.](https://code.visualstudio.com/)
1. [Clone the **epymorph** git repository.](https://github.com/NAU-CCL/Epymorph)
1. Open the project folder in VS Code.
1. Install the recommended extensions (which include Python, Jupyter, and ruff).
1. Open a terminal in VS Code (which should open a shell in the project folder) and run `uv sync`; this will create a virtual environment and install epymorph's dependencies (including dev dependencies). Additionally if you don't already have python installed, uv will install a suitable version for you.
1. VS Code may enable the new virtual environment automatically, but it may not. You can run the "Python: Select Interpreter" command from VS Code's command palette to be sure. You want to select the virtual environment in the project's ".venv" folder.

That's it! In the terminal, you can run the command `uv run epymorph --help` to check that epymorph has been successfully installed. You should see a description of epymorph's command-line interface.

The included VS Code settings configure the editor to run linting checks and auto-format the code on file save. You should verify that this is true! Make a trivial change to a python code file, like adding some blank lines, and save the file. The formatter should fix the file to remove the extraneous blank lines. If it doesn't, double check your configuration. We require a strict auto-formatter in order to maintain a clean and readable change history and avoid unnecessary merge conflicts in contributions.

## Configuration

epymorph developers using VS Code will find it convenient to use a `.env` file in the project root for providing configuration values (i.e., environment variables; as noted in the README).

For example, `USAGE.ipynb` fetches data from the US Census and you'll need to provide an API key for that to work.

First [request an API key from the Census](https://api.census.gov/data/key_signup.html). Once you have the key (a sequence of letters and numbers), create a file named `.env` in the root of the epymorph project containing these contents (for example, assuming our key is "abcd1234"):

```
CENSUS_API_KEY=abcd1234
```

You may need to reload VS Code first, but otherwise the values in this file will be used automatically. Of course you can set other environment variables here as well.

## Version release checklist

Pre-release tasks:

- Ensure all intended PRs/commits are merged to `main`
- Ensure your local main is up to date: `git pull origin main`
- Check for linting issues: `uv run ruff check --no-fix .`
- Check for formatting issues: `uv run ruff format --check`
- Check for data export issues: `uv run scripts/data_exports.py`
- Check for dependency issues: `uv run scripts/dep_check.py`
- Check fast tests: `uv run pytest tests/fast`
- Check slow tests: `uv run pytest tests/slow`
- Review docstrings for accuracy (especially anything new)
- Check for doc build issues: `uv run mkdocs build`
- Run numerical consistency checks including the release candidate branch

Release process:

- Create the version tag, e.g.: `git tag v1.2.0`
- Push the tag (to trigger the release workflow), e.g.: `git push origin v1.2.0`
- Monitor GitHub Actions release workflow (`release.yml`)
- Confirm the GitHub Release appears at https://github.com/NAU-CCL/Epymorph/releases
- Confirm the PyPI release appears at https://pypi.org/project/epymorph/
- Verify pip install epymorph==1.2.0 works in a clean environment (or trigger the installation matrix workflow)

Post release tasks:

- Edit the GitHub Release notes; the auto-generated notes are just a starting point
- Trigger a build and deploy of the documentation site to bring in the new API reference docs
- Announce the release!
