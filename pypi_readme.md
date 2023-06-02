[![Pre-Commit Status](https://github.com/xaviernogueira/Tabular_ML/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/xaviernogueira/Tabular_ML/actions/workflows/pre-commit.yml)
[![Tests Status](https://github.com/xaviernogueira/Tabular_ML/actions/workflows/tests.yml/badge.svg)](https://github.com/xaviernogueira/Tabular_ML/actions/workflows/tests.yml)
[![Coverage](https://codecov.io/gh/xaviernogueira/Tabular_ML/graph/badge.svg)](https://codecov.io/gh/xaviernogueira/Tabular_ML)

# `tabular_ml` - tabular machine learning simplified!
I've packaged and open sourced my personal machine learning tools to speed up your next data science project.

Train, evaluate, ensemble, and optimize hyperparameters from a standardized interface.

![repo_schematic](images/readme_image.png)

## Key Features
* Train models efficiently without worrying about library differences! `tabular_ml` implements library specific, performance oriented, patterns/classes under-the-hood (i.e., `xgboost.DMatrix -> xgboost.Booster`).
* Automate the K-Fold evaluation process across multiple models simultaneously (including ensembles).
* Rapidly optimize hyperparameters using [`optuna`](https://optuna.org/). Leverage our built-in parameter search spaces, or adjust to your needs.
* Plugin-able. Write your own plugins to extend functionality without forking (and consider contributing your plugins!).

**For full documentation see our GitHub ReadMe [here](https://github.com/xaviernogueira/Tabular_ML).**
