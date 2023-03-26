"""
LightGBM Regression.
"""
import lightgbm
import logging
import pandas as pd
import numpy as np
from optuna.trial import Trial
from typing import (
    Dict,
    Tuple,
    Optional,
    Any,
    List,
)
from ml_models.ml_model_shared import (
    MLModel,
    performance_scoring,
)


class LightGBMRegressionModel(MLModel):

    @staticmethod
    def train_model(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        model_params: Dict[str, Any],
        weights_train: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> lightgbm.Booster:

        # set up training weights
        if weights_train is not None:
            weights_train = weights_train.values

        train_data_ds = lightgbm.Dataset(
            data=x_train,
            label=y_train,
            weight=weights_train,
            categorical_feature=categorical_features,
        )

        evals = [
            (train_data_ds, 'train'),
        ]

        # train model and return log loss score for testing
        if 'early_stopping_rounds' in model_params.keys():
            early_stopping_rounds = model_params.pop('early_stopping_rounds')
        else:
            early_stopping_rounds = None
        if 'num_boost_round' in model_params.keys():
            num_boost_round = model_params.pop('num_boost_round')
        else:
            num_boost_round = None

        # return the trained model
        # TODO: get this in place correct!
        return lightgbm.train(
            params=model_params,
            train_set=train_data_ds,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            num_boost_round=num_boost_round,
        )

    @staticmethod
    def make_predictions(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        model_params: Dict[str, Any],
        weights_train: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> Tuple[lightgbm.Booster, np.ndarray]:

        # load in testing data
        test_data_ds = lightgbm.Dataset(
            data=x_test,
            categorical_feature=categorical_features,
        )

        # train the model
        lgbm_model = LightGBMRegressionModel.train_model(
            x_train,
            y_train,
            params=model_params.copy(),
            weights_train=weights_train,
            categorical_features=categorical_features,
        )

        # return prediction array
        return (
            lgbm_model,
            lgbm_model.predict(test_data_ds),
        )

    @staticmethod
    def objective(
        trial: Trial,
        features: pd.DataFrame,
        target: pd.Series,
        kfolds: int,
        metric_function: callable,
        weights: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None,
        random_state: Optional[int] = None,
    ) -> float:
        """
        LightGBM parameter search space for optuna.
        """

        # fill in more via https://catboost.ai/en/docs/references/training-parameters/common#bootstrap_type
        params = {
            #    'eval_metric': trial.suggest_categorical(
            #        'eval_metric', [
            #            'mae',
            #            # 'rmse',
            #        ]
            #    ),
            #    'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 10, 100),
            #    # lambda -> L2 regularization, default was 3
            #    'lambda': trial.suggest_int('lambda', 3, 8),
            #    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.4),
            #    'max_depth': trial.suggest_int('max_depth', 2, 8),
            #    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            #    'num_boost_round': trial.suggest_int('num_boost_round', 250, 1500),
        }

        logging.info(f'\n----------------------\n{params}')

        return performance_scoring(
            model=LightGBMRegressionModel,
            features=features,
            target=target,
            model_params=params,
            k_folds=kfolds,
            metric_function=metric_function,
            weights=weights,
            categorical_features=categorical_features,
            random_state=random_state,
        )
