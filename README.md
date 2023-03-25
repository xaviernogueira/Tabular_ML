Kaggle_Predicting-Parkinson-Disease-Progression
================================================
 A repo to store my attempt at the AMP Parkinson's Disease Progression Kaggle AI/ML competition. See https://www.kaggle.com/competitions/amp-parkinsons-disease-progression-prediction/overview

# To-Do
*** Enable custom loss function (i.e. SMAPE), and use in training (where possible).**
    * This should be passed in as a model_param, such that we can evaluate if training for SMAPE is even beneficial across K-Folds.
* Finish our concrete MLModel implementations.
* Do some EDA, without reinventing the wheel, and make public.
* Automate new logging files with a datetime stamp.
* Consider stratified K-Fold!
* Work on feature engineering.
* Set things up for GPU when possible.
* Use Kaggle `Gist` syntax to import our repo after making it public.

All together, with clever FE, optuna optimization overnight, and testing different random states we can get a great score.

# Contents
## `inputs` and `outputs`
Self explanatory. The `inputs` directory stores training data downloaded from [Kaggle](https://www.kaggle.com/competitions/amp-parkinsons-disease-progression-prediction/data). The `outputs` directory stores prediction output `.parquet` for quick reloading (as necessary...may not be used).

## `ml_models`
In order to enable experimentation, I defined  the Abstract Base Class `MLModel` in `ml_model_shared.py`, which allows all models implementing it's signature to be swapped interchangeably within K-Fold CV, model ensembles, and `optuna` hyperparameter optimization. 

Each concrete implementations of `MLModel` has the following:
* `train_model()` - returns a trained instance of the model after being passes X, and Y `pandas.DataFrame` and `pandas.Series` respectively. 
* `make_predictions()` - trains a model and returns predictions on a testing X `pandas.DataFrame` as a `numpy.array`.
* `objective()` - The name is misleading somewhat to align with `optuna` conventions. This function takes an `optuna.trial.Trial` as an argument and returns a performance score defined in `ml_model_shared.py`. The "meat" of the function defines the hyperparameter search space for a given model.

I implemented the following concrete implementations of `MLModel`, each stored in it's own sub-module:
* `ml_models.lasso_model.LassoLinearRegressor`
* `ml_models.catboost_model.CatBoostRegressor` - **done**
* `ml_models.xgboost_model.XGBoostRegressor`- **done**
* `ml_models.lgbm_model.LightGBMRegressor`
* `ml_models.svr_model.SupportVectorRegressor`
* `ml_models.nn_model.NeuralNetworkRegressor`

## `code_and_notebooks`

