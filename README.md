# epymorph

Prototype EpiMoRPH system written in Python. It is usable as a CLI program and also as a code library; for instance you may want to use it from within a Jupyter Notebook (see: `USAGE.ipynb`).

The `epymorph/data` directory is the model library, containing named implementations of IPMs, MMs, and GEOs. Ultimately our goal is to allow users to bring-their-own models by loading specification files, but for now they need to be registered in the model library.

The `doc/devlog` directory contains Jupyter Notebooks demonstrating features of epymorph and general development progress.

Beware: much of this code structure is experimental and subject to change!

## Project setup

For starters, you should have Python 3.11 installed and we'll assume it's accessible via the command `python3.11`.

You may need to install additional system packages for virtual environments and viewing plots. For example (on Ubuntu 22.04 LTS - jammy):

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

## Running from the command line

The most basic task epymorph can perform is to run a spatial, compartmental disease simulation and output the time-series data of compartment populations (prevalence) as well as new events (incidence).

A commonly-cited model was proposed by [Sen Pei, et al. in 2018](https://www.pnas.org/doi/10.1073/pnas.1708856115), modeling influenza in six southern US states. epymorph has an intra-population model (IPM), movement model (MM), and geographic model (GEO) that closely mimics Pei's experiment.

```bash
cd $PROJECT_DIRECTORY

# Activate the venv (if it's not already):
source .venv/bin/activate

# Prepare the simulation input file:
python -m epymorph prepare --ipm pei --mm pei --geo pei ./scratch/my-experiment.toml

# (./scratch is a convenient place to put temp files because our .gitignore excludes it)

# Now we need to edit the input file to specify the parameters needed by our combo of IPM and MM:
# (I'll use `cat` for this but you can use any text editor of course.)
cat << EOF >> ./scratch/my-experiment.toml
theta = 0.1
move_control = 0.9
infection_duration = 4.0
immunity_duration = 90.0
infection_seed_loc = 0
infection_seed_size = 10_000
EOF

# Now we can run the simulation:
python -m epymorph run ./scratch/my-experiment.toml --out ./scratch/output.csv

# Now if I open that csv file I see:
# - for each time-step (t) and population (p)
#   - prevalence data by compartment (c0, c1, c2)
#   - incidence data by event (e0, e1, e2)

# You can also run to display a chart:
python -m epymorph run ./scratch/my-experiment.toml --chart p0
```

To learn more about these and all other commands, you can always consult the CLI help:

```bash
python -m epymorph --help

# or for a specific subcommand
python -m epymorph run --help
```

### Other command-line tasks

Run all unit tests:

```bash
python -m unittest discover -v -s ./epymorph -p '*_test.py'
```

Run a simulation with the pdb debugger:

```bash
python -m pdb -m epymorph run ./scratch/my-experiment.toml
```

Profile the simulation and show the results in `snakeviz`:

```bash
TMP=$(mktemp /tmp/py-XXXXXXXX.prof)
python -m cProfile -o $TMP -m epymorph run ./scratch/my-experiment.toml --profile
snakeviz $TMP
```
