[![Pre-Commit Status](https://github.com/xaviernogueira/Tabular_ML/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/xaviernogueira/Tabular_ML/actions/workflows/pre-commit.yml)
[![Tests Status](https://github.com/xaviernogueira/Tabular_ML/actions/workflows/tests.yml/badge.svg)](https://github.com/xaviernogueira/Tabular_ML/actions/workflows/tests.yml)

# `tabular_ml` Documentation
I've open sourced my personal machine learning tools to speed up your next data science project!

Train, evaluate, ensemble, and optimize hyperparameters from a standardized interface.

![repo_schematic](images/readme_image.png)

## Overview

## Currently supported models

## Evaluate

## Optimize hyperparameters

## Contribute
## `/tabular_ml.ml_models`
In order to enable experimentation, I defined  the Abstract Base Class `MLModel` in `ml_model_shared.py`, which allows all models implementing it's signature to be swapped interchangeably within K-Fold CV, model ensembles, and `optuna` hyperparameter optimization.

Each concrete implementations of `MLModel` has the following:
* `train_model()` - returns a trained instance of the model after being passes X, and Y `pandas.DataFrame` and `pandas.Series` respectively.
* `train_and_predict()` - trains a model and returns predictions on a testing X `pandas.DataFrame` as a `numpy.array`.
* `objective()` - The name is misleading somewhat to align with `optuna` conventions. This function takes an `optuna.trial.Trial` as an argument and returns a performance score defined in `ml_model_shared.py`. The "meat" of the function defines the hyperparameter search space for a given model.

In `ml_model_shared` I also stored evaluation functions that accept any concrete implementation of `MLModel`:
* `k_fold_cv()` runs K-Fold Cross Validation and stores all relevant outputs including model instances in a `KFoldOutput` python `dataclasses.dataclass` instance.
* `find_optimal_parameters()` and `performance_scoring()` work together to enable an `optuna` search of the hyperaparameter space with a given scoring metric. Note that the mean K-Fold score +/- (depending on metric polarity) the standard deviation between folds is used.

I implemented the following concrete implementations of `MLModel`, each stored in it's own sub-module:

**CatBoost:**
* `catboost_model.CatBoostRegressionModel`
* `catboost_model.CatBoostClassificationModel`

**XGBoost:**
* `xgboost_model.XGBoostRegressionModel`

**LightGBM:**
* `lgbm_model.LightGBMRegressionModel`

**Sklearn Linear Models:**
* `linear_models.LinearRegressionModel`
* `linear_models.RidgeRegressionModel`
* `linear_models.LassoRegressionModel`
* `linear_models.ElasticNetRegressionModel`
* `linear_models.BayesianRidgeRegressionModel`

**Support Vector Models:**
* `svr_model.SupportVectorRegressionModel` (not implemented)

**Neural Network Models:**
* `nn_model.NeuralNetworkRegressionModel` (not implemented)
