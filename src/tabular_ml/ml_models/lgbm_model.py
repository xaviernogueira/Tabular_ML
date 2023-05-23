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
from tabular_ml.functions import (
    performance_scoring,
)
from tabular_ml.base import (
    MLModel,
    ModelTypes,
    OptunaRangeDict,
)
from tabular_ml.factory import ModelFactory

# TODO: Finish this implementation


class BaseLightGBMModel(MLModel):

    model_type: ModelTypes = None
    optuna_param_ranges: OptunaRangeDict = None

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
            # TODO: see if we can do dictionary kwargs instead, for now leave as default
            num_boost_round = 100

        # return the trained model
        # TODO: get this in place correct!
        return lightgbm.train(
            params=model_params,
            train_set=train_data_ds,
            # TODO: get evals working evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            num_boost_round=num_boost_round,
        )

    @staticmethod
    def make_predictions(
        trained_model: lightgbm.Booster,
        x_test: pd.DataFrame,
        categorical_features: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Makes predictions with LightGBM"""

        # TODO: figure out why this doesn't work
        # load in testing data in a library optimized way
        # test_data_ds = lightgbm.Dataset(
        #    data=x_test,
        #    categorical_feature=categorical_features,
        # )

        return trained_model.predict(
            x_test,
            categorical_features=categorical_features,
        )

    @classmethod
    def train_and_predict(
        cls,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        model_params: Dict[str, Any],
        weights_train: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> Tuple[lightgbm.Booster, np.ndarray]:

        # get trained model
        lgbm_model = cls.train_model(
            x_train,
            y_train,
            model_params=model_params.copy(),
            weights_train=weights_train,
            categorical_features=categorical_features,
        )

        # return prediction array
        return (
            lgbm_model,
            cls.make_predictions(
                trained_model=lgbm_model,
                x_test=x_test,
                categorical_features=categorical_features,
            ),
        )

    @classmethod
    def objective(
        cls,
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

        function_mapping = {
            'regression': 'regression',
            'classification': 'binary',
        }
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
            model=cls,
            features=features,
            target=target,
            model_params=params,
            k_folds=kfolds,
            metric_function=metric_function,
            weights=weights,
            categorical_features=categorical_features,
            random_state=random_state,
        )


# @ModelFactory.implemented_model
class LightGBMRegressionModel(BaseLightGBMModel):
    model_type: ModelTypes = 'regression'
    optuna_param_ranges: OptunaRangeDict = {
        'objective': ['reg:squarederror'],
        'eval_metric': ['mae'],
        'early_stopping_rounds': (10, 100),
        'lambda': (3, 8),
        'learning_rate': (0.01, 0.4),
        'max_depth': (2, 8),
        'colsample_bytree': (0.5, 1.0),
        'num_boost_round': (250, 1500),
    }


# @ModelFactory.implemented_model
class LightGBMClassificationModel(BaseLightGBMModel):
    model_type: ModelTypes = 'classification'
    optuna_param_ranges: OptunaRangeDict = {
        'objective': ['reg:squarederror'],
        'eval_metric': ['mae'],
        'early_stopping_rounds': (10, 100),
        'lambda': (3, 8),
        'learning_rate': (0.01, 0.4),
        'max_depth': (2, 8),
        'colsample_bytree': (0.5, 1.0),
        'num_boost_round': (250, 1500),
    }
