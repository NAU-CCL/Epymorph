# epymorph

Prototype EpiMoRPH system written in Python for exploring object-oriented design work.

The primary entry point to the program (currently) is `__main__.py` with a number of available subcommands. Example commands are given below, or you can consult the command help in the console.

System submodules include `epi.py` (IPM), `geo.py` (GeoM), and `movement.py` (MM). `simulation.py` brings all of these together in an execution loop (RUME) to produce incidence and prevalence output.

The `epymorph/data` directory is the model library, containing named implementations of IPMs, MMs, and Geos.

The `doc/devlog` directory contains Jupyter notebooks demonstrating features of epymorph and general development progress.

Beware: much of this code structure is experimental and subject to change!

## Project setup

For starters, you should have Python 3.11 installed and we'll assume it's accessible via the command `python3.11`.

You will need system packages for virtual environments and viewing plots. For example (tested on Ubuntu 22.04 LTS - jammy):

```bash
sudo apt install python3.11-venv python3.11-tk
```

Using VS Code, install the project's recommended IDE extensions. Then use the "Python - Create Environment" command (`Ctrl+Shift+P`) to create a Venv environment and install the modules from `requirements.txt` and `requirements-dev.txt`.

Or you can set up from the command line:

```bash
cd $PROJECT_DIRECTORY
# create the virtual environment
python3.11 -m venv .venv
# activate it
source .venv/bin/activate
# then install the requirements
python3.11 -m pip install -r requirements.txt
python3.11 -m pip install -r requirements-dev.txt
```

## Running the project

There are VS Code tasks configured for running (or profiling) the currently open file, and its Testing view should enable running the unit tests.

Alternatively, from the command line (making sure you've activated the venv):

```bash
# Running the main program:
python3 -m epymorph run --ipm pei --mm pei --geo pei --params ~/my-params.toml --start_date 2010-01-01 --duration 150d --chart e0

# Getting general command help:
python3 -m epymorph --help
# Getting help on a specific subcommand:
python3 -m epymorph run --help

# Running all unit tests:
python3 -m unittest discover -v -s ./epymorph -p '*_test.py'

# Profiling the main program and opening the results in snakeviz:
TMP=$(mktemp /tmp/py-XXXXXXXX.prof); python3 -m cProfile -o $TMP -m epymorph sim pei_py --profile; snakeviz $TMP
```
