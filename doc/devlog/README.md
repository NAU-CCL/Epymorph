# devlog

This folder is a handy place to put Jupyter notebooks or other documents which help to demonstrate the state of the project at a particular point in time.

Checking in the .ipynb file is sufficient, as long as it has been fully rendered. (GitHub even knows how to display them as-is!) But because they are point-in-time demos, there's no guarantee they will run using future versions of the code. So please include in the document the git commit hash being demonstrated. (The idea being that a developer should be able to check out that version of the project if they needed to re-run/export the notebook.)

## Export

Jupyter notebooks can be exported to HTML if needed using VS Code (`CTRL+SHIFT+P, Jupyter: Export to HTML`) or from the command line, for example:

```bash
# make sure .venv is activated
python -m jupyter nbconvert ./doc/devlog/2023-05-03.ipynb --to html --output /home/tcoles/Workspaces/Epymorph/doc/devlog/2023-05-03.html
```
