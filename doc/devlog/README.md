# devlog

This folder is a handy place to put Jupyter notebooks or other documents which help to demonstrate the state of the project at a particular point in time.

Because they are point-in-time demos, there's no guarantee they will compile on future versions of the code. So please check in the source file as well as an HTML-rendered version (for things like Jupyter notebooks), and include in the document the git commit hash being demonstrated. (The idea being that a developer should be able to check out that version of the project if they needed to re-export the notebook.)

Jupyter notebooks can be exported to HTML using VS Code (`CTRL+SHIFT+P, Jupyter: Export to HTML`) or from the command line, for example:

```bash
# make sure .venv is activated
python -m jupyter nbconvert ./doc/devlog/2023-05-03.ipynb --to html --output /home/tcoles/Workspaces/Epymorph/doc/devlog/2023-05-03.html
```

Do export to a file named the same as the source file (besides the extension of course).

Exporting to HTML is preferred over PDF because:
- makes sense to standardize on one thing,
- paginated code samples make me sad,
- git-friendliness (text vs. binary), and
- `nbconvert` requires system packages to export to PDF and I'm too lazy to document the setup steps.
