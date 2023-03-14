# epymorph

Prototype EpiMoRPH system written in Python for exploring object-oriented design work.

## Project setup

For starters, you should have Python 3 installed and we'll assume it's accessible via the command `python3`.

You will need system packages for virtual environments, viewing plots, and (optionally) viewing profiling charts. For example:

```bash
sudo apt install python3-venv python3-tk
python3 -m pip install snakeviz
```

Using VS Code, install the project's recommended extensions. Then use the "Python - Create Environment" command (`Ctrl+Shift+P`) to create a Venv environment and install the modules from `requirements.txt`.

Or you can set up from the command line:

```bash
cd $PROJECT_DIRECTORY
# create the virtual environment
python3 -m venv .venv
# activate it
source .venv/bin/activate
# then install the requirements
python3 -m pip install -r requirements.txt
```
## Running the project

There are VS Code tasks configured for running (or profiling) the currently open file, and its Testing view should enable running the unit tests.

Alternatively, from the command line:

```bash
# Running the main program:
python3 ./src/main.py

# Running all unit tests:
python3 -m unittest discover -v -s ./src -p '*_test.py'

# Profiling the main program and opening the results in snakeviz:
TMP=$(mktemp /tmp/py-XXXXXXXX.prof); python3 -m cProfile -o $TMP ./src/main.py --profile; snakeviz $TMP
```
