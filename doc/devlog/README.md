# devlog

This folder is a handy place to put Jupyter notebooks or other documents which help to demonstrate the state of the project at a particular point in time. Because these are point-in-time demos, there's no guarantee they will continue to be runnable as changes are made to the project. That's expected. (If you really need to run a Notebook in its original context, you can git-checkout back to the commit that added the Notebook and run it there; caveat emptor).

**Notable Devlogs:** Some devlogs are more than mere experiments, acting like documentation -- demoing new features of the project or novel approaches to problem solving. These devlogs are check-marked as _Maintained_ in the index below (indicating that they are intended to be kept up to date as the project changes).

## Devlog Index

| Filename | Author | Maintained? | Description |
| --- | --- | --- | --- |
| 2023-05-03.ipynb | Tyler | | An obsolete demo, running different movement models with the Pei IPM/Geo. |
| 2023-05-04.ipynb | Tyler | | An experiment in running epymorph simulations in parallel using `multiprocessing`.
| 2023-05-16.ipynb | Tyler | | An attempt to simplify project organization by turning then-hard-coded examples into Jupyter notebooks. |
| 2023-05-17.ipynb | Tyler | | Demo of the newly-added .toml file input scheme for providing parameter values for CLI simulations. |
| 2023-05-31-mm.ipynb | Alex | | Movement model experimentation. |
| 2023-05-31-sirh.ipynb | Alex | | SIRH IPM experimentation. |
| 2023-06-01-pei-example.ipynb | Frank | | Demonstration of the 'pei' movement model. |
| 2023-06-01-sparsemod-example.ipynb | Frank | | Demonstration of the 'sparsemod' movement model. |
| 2023-06-28.ipynb | Tyler | | Proving validity of the newly-added declarative compartment model IPM implementation. |
| 2023-06-30.ipynb | Tyler | ✓ | Demonstrating the newly-added declarative compartment model IPM system. (This is a good reference for building custom IPMs, so we're keeping it current.) |
| 2023-07-06.ipynb | Tyler | ✓ | Creates the Pei Geo. (Maintained until such a time as the ADRIO system can replace it.) |
| 2023-07-07.ipynb | Tyler | ✓ | Creates the 2015 US States and US Counties Geos. (Maintained until such a time as the ADRIO system can replace them.) |
| 2023-07-12.ipynb | Tyler | ✓ | Creates the 2019 Maricopa County CBGs Geo. (Maintained until such a time as the ADRIO system can replace it.) |
| 2023-07-13.ipynb | Tyler | ✓ | Implements a compatibility matrix test: are all possible combinations of IPM/MM/GEO valid? |
| 2023-07-14.ipynb | Tyler | | Demonstrates filtering a geo down to a subset of its nodes. (While the motivation to do this still exists, recent changes have made this exact approach obsolete.) |
| 2023-07-20-movement-probs.ipynb | Tyler | | Analyzing statistical correctness of our movement processing algorithms. |
| 2023-07-24.ipynb | Tyler | | Experiments with adapting an IPM by "attaching" a function to an IPM parameter. This approach has been superseded by a design for direct support for functional parameters. |
| 2023-08-11.ipynb | Tyler | | Demonstrates performance differences between the Basic and Hypercube movement engines. A pending refactor will make the concept of movement engines obsolete. |
| 2023-08-17.ipynb | Tyler | ✓ | Demonstrates the newly-added Initializer functions, including library examples and custom initializers. |
| 2023-08-23.ipynb | Tyler | ✓ | Describes what IPM attribute broadcasting is and why it's useful. Introduces our concept of data shapes. |
| 2023-09-22-adrio-demo.ipynb | Trevor | | Demonstrates the newly-added ADRIOs functionality by fetching data from the US Census ACS5. |
| 2023-09-29.ipynb | Tyler | | Experiments in representing exogenous births and deaths in the existing compartment IPM system. |
| 2023-10-05.ipynb | Tyler | | Demonstrates the newly-added first-class support for exogenous births and deaths in the compartment IPM system. |
| 2023-10-10.ipynb | Tyler | | A demo of various epymorph workflows in a Notebook environment, designed for a live presentation. |
| 2023-10-26.ipynb | Tyler | | Describes a major Geo system refactor and introduces new systems. |
| 2023-11-03-seirs-example.ipynb | Ajay | | Demonstrates the building and running of an SEIRS model. |
| 2023-11-08.ipynb | Ajay | | Demonstration of using proxy geo to access data in parameter functions. |
| 2023-11-08-age-ipm.ipynb | Jarom | | Initial prototyping of age-class IPMs. |
| 2023-11-15.ipynb | Ajay | | Detailed description of parameter functions functionality. |
| 2023-11-20-adrio-phase-2-demo.ipynb | Trevor | | Demonstrates the refactor work on DynamicGeos and the ADRIO system, and geo cache handling. |
| 2023-11-22-ipm-probs.ipynb | Tyler | | Analyzing statistical correctness of our IPM processing algorithms. |
| 2023-12-05.ipynb | Tyler | | A brief tour of changes to epymorph due to the refactor effort. |
| 2024-01-08.ipynb | Tyler | | Another functional parameters demonstration, revisiting the Bonus Example from 2023-10-10. |
| 2024-02-06-adrio-demo.ipynb | Trevor | | Demonstrates the ADRIO system using code updated for latest changes. |
| 2024-02-06.ipynb | Tyler | | Revisiting age-class IPMs, and thinking about modularity of approach. |
| 2024-02-12.ipynb | Tyler | | Continued age-class IPM work, this time in more than one geo node. |
| 2024-02-14.ipynb | Tyler | | Prep work related to the "Z-virus" workshop. (Not very organized.) |
| 2024-03-01.ipynb | Tyler | | Getting the indices of IPM events and compartments by name with wildcard support. |
| 2024-03-13.ipynb | Tyler | | Showing off movement data collection (NEW!) |
| 2024-04-04-draw-demo.ipynb | Izaac | | Showing the new draw module for visualising IPMs (NEW!) |
| 2024-04-16.ipynb | Izaac | | Showing error handling for common ipm errors (NEW!)|
| 2024-04-25.ipynb | Tyler | | Integration test: epymorph cache utilities |
| 2024-05-03.ipynb | Tyler | | Integration test: loading US Census geography from TIGER |

## Contributing

When adding a devlog, checking in the .ipynb file is sufficient as long as it has been fully rendered. (GitHub even knows how to display them as-is!)

## Export

Jupyter notebooks can be exported to HTML if needed using VS Code (`CTRL+SHIFT+P, Jupyter: Export to HTML`) or from the command line, for example:

```bash
# make sure .venv is activated
python -m jupyter nbconvert ./doc/devlog/2023-05-03.ipynb --to html --output <your_absolute_path>/Epymorph/doc/devlog/2023-05-03.html
```
