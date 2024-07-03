# epymorph

Prototype EpiMoRPH system written in Python. It is usable as a code library, for instance, from within a Jupyter Notebook (see: `USAGE.ipynb`).

The `epymorph/data` directory is the model library, containing named implementations of IPMs, MMs, and GEOs. Ultimately our goal is to allow users to bring their own models by loading specification files, but for now they need to be registered in the model library.

The `doc/devlog` directory contains Jupyter Notebooks demonstrating features of epymorph and general development progress.

Beware: much of this code structure is experimental and subject to change!

## Basic usage

See the `USAGE.ipynb` Notebook for a simple usage example.

## Development setup

For starters, you should have Python 3.11 installed and we'll assume it's accessible via the command `python3.11`.

You may need to install additional system packages for virtual environments and viewing plots. For example (on Ubuntu 22.04 LTS - jammy):

```bash
sudo apt install python3.11-venv python3.11-tk
```

If you are using VS Code, install the project's recommended IDE extensions. Then use the "Python - Create Environment" command (`Ctrl+Shift+P`) to create a Venv environment and install all dependencies (including `dev`).

Or you can set up from the command line:

```bash
cd $PROJECT_DIRECTORY

# create the virtual environment
python3.11 -m venv .venv

# activate it (after which `python` should be bound to the venv python)
# NOTE: activating venv on Windows is different; see documentation
source .venv/bin/activate

# then install the project in editable mode
python -m pip install --editable ".[dev]"
```

Make sure you have correctly configured auto-formatting in your development environment. We're currently using autopep8 and isort. These formatting tools should run every time you save a file.

### Other command-line tasks

Run all unit tests:

```bash
python -m unittest discover -v -s ./epymorph -p '*_test.py'
```
