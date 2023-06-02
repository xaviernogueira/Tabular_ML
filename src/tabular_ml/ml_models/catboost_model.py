"""
CatBoost regression.
"""
import catboost
import warnings
import logging
import pandas as pd
import numpy as np
from optuna.trial import Trial
from typing import (
    Dict,
    List,
    Tuple,
    Optional,
    Union,
    Any,
)
from tabular_ml.functions import (
    performance_scoring,
)
from tabular_ml.base import (
    MLModel,
    ModelTypes,
    OptunaRangeDict,
)
from tabular_ml.utilities import (
    get_optuna_ranges,
    suggest_optuna_params,
)
from tabular_ml.factory import ModelFactory

CatBoostModels = Union[
    catboost.CatBoostRegressor,
    catboost.CatBoostClassifier,
]


class BaseCatBoostModel(MLModel):
    """Hold shared functions for CatBoost models.

    NOTE: This cannot be before being inherited into an MLModel class!
    """

    model_type: ModelTypes = None

    @classmethod
    def train_model(
        cls,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        model_params: Dict[str, Any],
        weights_train: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> CatBoostModels:
        """Trains a CatBoostRegressor model"""

        # prep training weights
        if weights_train is not None:
            weights_train = weights_train.values

        # make pool to load train data
        train_data_pool = catboost.Pool(
            data=x_train,
            label=y_train,
            weight=weights_train,
            cat_features=categorical_features,
        )

        # return the trained model
        catboost_model = cls.model_object(**model_params)
        return catboost_model.fit(
            train_data_pool,
            verbose=False,
        )

    @staticmethod
    def make_predictions(
        trained_model: CatBoostModels,
        x_test: pd.DataFrame,
        categorical_features: Optional[List[str]] = None,
    ) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def train_and_predict(
        cls: MLModel,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        model_params: Dict[str, Any],
        weights_train: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> Tuple[CatBoostModels, np.ndarray]:
        """Trains a CatBoost model and makes predictions.

        NOTE: This cannot be before being inherited into an MLModel class!
        """

        # train a model
        catboost_model = cls.train_model(
            x_train,
            y_train,
            model_params,
            weights_train,
            categorical_features,
        )

        # return predictions array
        return (
            catboost_model,
            cls.make_predictions(
                trained_model=catboost_model,
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
        custom_optuna_ranges: Optional[OptunaRangeDict] = None,
    ) -> float:
        """
        CatBoost parameter search space for optuna.
        """
        # get parameter ranges
        param_ranges = get_optuna_ranges(
            cls.optuna_param_ranges,
            custom_optuna_ranges=custom_optuna_ranges,
        )

        # set up parameters
        function_mapping = {
            'learning_rate': trial.suggest_float,
            'early_stopping_rounds': trial.suggest_int,
            'depth': trial.suggest_int,
            'bootstrap_type': trial.suggest_categorical,
            'colsample_bylevel': trial.suggest_float,
            'bagging_temperature': trial.suggest_float,
            'subsample': trial.suggest_float,
        }
        params = suggest_optuna_params(
            trial,
            param_ranges,
            function_mapping,
        )

        # trim parameters based on task type
        if 'task_type' in params.keys() and 'colsample_bylevel' in params.keys():
            if params['task_type'] == 'GPU':
                del params['colsample_bylevel']

        # trim parameters based on bootstrap_type type
        if 'bootstrap_type' in params.keys():
            if params['bootstrap_type'] != 'Bayesian':
                try:
                    del params['bagging_temperature']
                except KeyError:
                    pass

            if params['bootstrap_type'] != 'Bernoulli':
                try:
                    del params['subsample']
                except KeyError:
                    pass

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


@ModelFactory.implemented_model
class CatBoostRegressionModel(BaseCatBoostModel):

    model_type: ModelTypes = 'regression'
    model_object: CatBoostModels = catboost.CatBoostRegressor
    optuna_param_ranges: OptunaRangeDict = {
        'objective': ['MAE', 'RMSE'],
        'bootstrap_type': ['Bayesian', 'Bernoulli', 'MVS'],
        'subsample': (0.1, 1),
        'learning_rate': (0.01, 0.3),
        'early_stopping_rounds': (10, 100),
        'depth': (2, 10),
        'colsample_bylevel': (0.01, 0.1),
        'l2_leaf_reg': (3, 8),
        'iterations': (500, 2000),
    }

    @staticmethod
    def make_predictions(
        trained_model: catboost.CatBoostRegressor,
        x_test: pd.DataFrame,
        categorical_features: Optional[List[str]] = None,
    ) -> np.ndarray:
        # load in testing data
        test_data_pool = catboost.Pool(
            data=x_test,
            cat_features=categorical_features,
        )
        return trained_model.predict(test_data_pool)


@ModelFactory.implemented_model
class CatBoostClassificationModel(BaseCatBoostModel):

    model_type: ModelTypes = 'classification'
    model_object: CatBoostModels = catboost.CatBoostClassifier
    optuna_param_ranges: OptunaRangeDict = {
        'learning_rate': (0.01, 0.3),
        'early_stopping_rounds': (10, 100),
        'depth': (2, 10),
        'bootstrap_type': ['Bayesian', 'Bernoulli', 'MVS'],
        'colsample_bylevel': (0.01, 0.1),
        'bagging_temperature': (0, 10),
        'subsample': (0.1, 1),
    }

    @staticmethod
    def make_predictions(
        trained_model: catboost.CatBoostClassifier,
        x_test: pd.DataFrame,
        categorical_features: Optional[List[str]] = None,
    ) -> np.ndarray:
        # load in testing data
        test_data_pool = catboost.Pool(
            data=x_test,
            cat_features=categorical_features,
        )
        return trained_model.predict_proba(test_data_pool)
