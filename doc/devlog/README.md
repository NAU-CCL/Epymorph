# devlog

This folder is a handy place to put Jupyter notebooks or other documents which help to demonstrate the state of the project at a particular point in time.

Because these are point-in-time demos, there's no guarantee they will continue to be runnable as changes are made to the project. That's expected. If you really need to run a Notebook in its original context, you should be able to git-checkout back to the git commit that added the Notebook and run it there (caveat emptor).

## Notable devlogs

Some devlogs are experiments, but some act like documentation of a sort -- demoing new features of the project. Here's an index of the most historically interesting devlogs.

- **2023-06-30**: demos how to write IPMs
- **2023-07-12**, **2023-07-07**, **2023-07-06**: are scripts to generate GEOs, sometimes loading data from third-party APIs
- **2023-07-13**: runs a compatibility matrix test to determine which IPM/MM/GEO combos run successfully (and how long they take to run)
- **2023-08-17**: demos simulation Initializer functions for establishing the initial compartments of a simulation
- **2023-08-23**: demos how parameter broadcasting can allow users to provide multiple shapes of data for simulation parameters
- **2023-10-05**: demos creating an IPM with exogenous inputs (births and deaths external to the model)

## Contributing

When adding a devlog, checking in the .ipynb file is sufficient as long as it has been fully rendered. (GitHub even knows how to display them as-is!)

## Export

Jupyter notebooks can be exported to HTML if needed using VS Code (`CTRL+SHIFT+P, Jupyter: Export to HTML`) or from the command line, for example:

```bash
# make sure .venv is activated
python -m jupyter nbconvert ./doc/devlog/2023-05-03.ipynb --to html --output <your_absolute_path>/Epymorph/doc/devlog/2023-05-03.html
```
