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
from tabular_ml.utilities import get_optuna_ranges
from tabular_ml.factory import ImplementedModel


@ImplementedModel
class CatBoostRegressionModel(MLModel):

    model_type: ModelTypes = 'regression'
    optuna_param_ranges: OptunaRangeDict = {
        'objective': ['MAE', 'RMSE'],
        'learning_rate': (0.01, 0.3),
        'early_stopping_rounds': (10, 100),
        'depth': (2, 10),
        'colsample_bylevel': (0.01, 0.1),
        'l2_leaf_reg': (3, 8),
        'iterations': (500, 2000),
    }

    @staticmethod
    def train_model(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        model_params: Dict[str, Any],
        weights_train: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> catboost.CatBoostRegressor:
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
        catboost_model = catboost.CatBoostRegressor(**model_params)
        return catboost_model.fit(
            train_data_pool,
            verbose=False,
        )

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

    @classmethod
    def train_and_predict(
        cls,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        model_params: Dict[str, Any],
        weights_train: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> Tuple[catboost.CatBoostRegressor, np.ndarray]:
        """Trains a CatBoost model and makes predictions"""

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

        # fill in more via https://catboost.ai/en/docs/references/training-parameters/common#bootstrap_type
        params = {
            'objective': trial.suggest_categorical(
                'objective',
                param_ranges['objective'],
            ),
            'learning_rate': trial.suggest_float(
                'learning_rate',
                param_ranges['learning_rate'][0],
                param_ranges['learning_rate'][-1],
            ),
            'early_stopping_rounds': trial.suggest_int(
                'early_stopping_rounds',
                param_ranges['early_stopping_rounds'][0],
                param_ranges['early_stopping_rounds'][-1],
            ),
            'depth': trial.suggest_int(
                'depth',
                param_ranges['depth'][0],
                param_ranges['depth'][-1],
            ),
            'colsample_bylevel': trial.suggest_float(
                'colsample_bylevel',
                param_ranges['colsample_bylevel'][0],
                param_ranges['colsample_bylevel'][-1],
            ),
            'l2_leaf_reg': trial.suggest_float(
                'l2_leaf_reg',
                param_ranges['l2_leaf_reg'][0],
                param_ranges['l2_leaf_reg'][-1],
            ),
            'iterations': trial.suggest_int(
                'iterations',
                param_ranges['iterations'][0],
                param_ranges['iterations'][-1],
            ),
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


@ImplementedModel
class CatBoostClassificationModel(MLModel):

    model_type: ModelTypes = 'classification'
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
    def train_model(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        model_params: Dict[str, Any],
        weights_train: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> catboost.CatBoostClassifier:
        """Trains a CatBoostClassifier model"""

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
        catboost_model = catboost.CatBoostClassifier(**model_params)
        return catboost_model.fit(
            train_data_pool,
            verbose=False,
        )

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
        class_idx = list(trained_model.classes_).index(1)
        return trained_model.predict_proba(test_data_pool)[:, class_idx]

    @classmethod
    def train_and_predict(
        cls,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        model_params: Dict[str, Any],
        weights_train: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> Tuple[catboost.CatBoostClassifier, np.ndarray]:
        """Trains a XGBoost model and makes predictions"""

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

        # TODO: support more parameters
        # https://catboost.ai/en/docs/references/training-parameters/common#bootstrap_type
        params = {
            'learning_rate': trial.suggest_float(
                'learning_rate',
                param_ranges['learning_rate'][0],
                param_ranges['learning_rate'][-1],
            ),
            'early_stopping_rounds': trial.suggest_int(
                'early_stopping_rounds',
                param_ranges['early_stopping_rounds'][0],
                param_ranges['early_stopping_rounds'][-1],
            ),
            'depth': trial.suggest_int(
                'depth',
                param_ranges['depth'][0],
                param_ranges['depth'][-1],
            ),
            'bootstrap_type': trial.suggest_categorical(
                'bootstrap_type',
                param_ranges['bootstrap_type'],
            ),
        }

        if 'task_type' in params.keys():
            if not params['task_type'] == 'GPU':
                params['colsample_bylevel'] = trial.suggest_float(
                    'colsample_bylevel',
                    param_ranges['colsample_bylevel'][0],
                    param_ranges['colsample_bylevel'][-1],
                )
        if params['bootstrap_type'] == 'Bayesian':
            params['bagging_temperature'] = trial.suggest_float(
                'bagging_temperature',
                param_ranges['bagging_temperature'][0],
                param_ranges['bagging_temperature'][-1],
            )
        elif params['bootstrap_type'] == 'Bernoulli':
            params['subsample'] = trial.suggest_float(
                'subsample',
                param_ranges['subsample'][0],
                param_ranges['subsample'][-1],
            )

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
