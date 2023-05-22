[![Pre-Commit Status](https://github.com/xaviernogueira/Tabular_ML/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/xaviernogueira/Tabular_ML/actions/workflows/pre-commit.yml)
[![Tests Status](https://github.com/xaviernogueira/Tabular_ML/actions/workflows/tests.yml/badge.svg)](https://github.com/xaviernogueira/Tabular_ML/actions/workflows/tests.yml)

# `tabular_ml` - tabular machine learning simplified!
I've packaged and open sourced my personal machine learning tools to speed up your next data science project.

Train, evaluate, ensemble, and optimize hyperparameters from a standardized interface.

![repo_schematic](images/readme_image.png)

## Key Features
* Train models efficiently without worrying about library differences! `tabular_ml` implements library specific, performance oriented, patterns/classes under-the-hood (i.e., `xgboost.DMatrix -> xgboost.Booster`).
* Automate the K-Fold evaluation process across multiple models simultaneously (including ensembles).
* Rapidly optimize hyperparameters using [`optuna`](https://optuna.org/). Leverage our built-in parameter search spaces, or adjust to your needs.
* Behavior you can trust long term. We use a type-checking "factory" design pattern and robust test suite to provide reliability as the library grows over time.


# Library Documentation

## Getting started
This library is available on PyPI and can be easily pip installed into your environment.
```
pip install tabular_ml
```
## Currently supported models
**[`catboost`](https://catboost.ai/en/docs/)**
* `CatBoostRegressionModel`
* `CatBoostClassificationModel`

**[`xgboost`](https://xgboost.readthedocs.io/en/stable/python/index.html)**
* `XGBoostRegressionModel`

**[`lightgbm`](https://lightgbm.readthedocs.io/en/v3.3.2/)**
* `LightGBMRegressionModel`

**[`sklearn.linear_models`](https://scikit-learn.org/stable/modules/linear_model.html)**
* `LinearRegressionModel`
* `RidgeRegressionModel`
* `LassoRegressionModel`
* `ElasticNetRegressionModel`
* `BayesianRidgeRegressionModel`

## Evaluate

## Optimize hyperparameters

## Contribute
