"""
CatBoost regression.
"""
import catboost
import logging
import pandas as pd
import numpy as np
from optuna.trial import Trial
from typing import (
    Dict,
    List,
    Optional,
    Any,
)
from ml_models.ml_model_shared import (
    MLModel,
    performance_scoring,
)


class CatBoostRegressionModel(MLModel):

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

        # TODO: enable custom eval function!

        # return the trained model
        catboost_model = catboost.CatBoostRegressor(**model_params)
        return catboost_model.fit(
            train_data_pool,
            verbose=False,
        )

    @staticmethod
    def make_predictions(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        model_params: Dict[str, Any],
        weights_train: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Trains a XGBoost model and makes predictions"""
        # load in testing data
        test_data_pool = catboost.Pool(
            data=x_test,
            cat_features=categorical_features,
        )

        # train a model
        catboost_model = CatBoostRegressionModel.train_model(
            x_train,
            y_train,
            model_params,
            weights_train,
            categorical_features,
        )

        # return predictions array
        return catboost_model.predict(test_data_pool)

    @staticmethod
    def objective(
        trial: Trial,
        features: pd.DataFrame,
        target: pd.Series,
        kfolds: int,
        metric_function: callable,
        weights: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None,
        peaks_indices: Optional[List[int]] = None,
        random_state: Optional[int] = None,
    ) -> float:
        """
        CatBoost parameter search space for optuna.
        """

        # fill in more via https://catboost.ai/en/docs/references/training-parameters/common#bootstrap_type
        params = {
            'objective': trial.suggest_categorical(
                'objective', [
                    'MAE',
                    # 'RMSE',
                ]
            ),
            'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 10, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 3, 8),
            'depth': trial.suggest_int('depth', 2, 8),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.01, 0.4),
            'iterations': trial.suggest_int('iterations', 500, 2000),
        }

        logging.info(f'\n----------------------\n{params}')

        return performance_scoring(
            model=CatBoostRegressionModel,
            features=features,
            target=target,
            peaks_indices=peaks_indices,
            model_params=params,
            k_folds=kfolds,
            metric_function=metric_function,
            weights=weights,
            categorical_features=categorical_features,
            random_state=random_state,
        )


class CatBoostClassificationModel(MLModel):

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
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        model_params: Dict[str, Any],
        weights_train: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Trains a XGBoost model and makes predictions"""
        # load in testing data
        test_data_pool = catboost.Pool(
            data=x_test,
            cat_features=categorical_features,
        )

        # train a model
        catboost_model = CatBoostClassificationModel.train_model(
            x_train,
            y_train,
            model_params,
            weights_train,
            categorical_features,
        )

        # return predictions array
        return catboost_model.predict(test_data_pool)[:, 1]

    @staticmethod
    def objective(
        trial: Trial,
        features: pd.DataFrame,
        target: pd.Series,
        kfolds: int,
        metric_function: callable,
        weights: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None,
        peaks_indices: Optional[List[int]] = None,
        random_state: Optional[int] = None,
    ) -> float:
        """
        CatBoost parameter search space for optuna.
        """

        # fill in more via https://catboost.ai/en/docs/references/training-parameters/common#bootstrap_type
        params = {
            # "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1), does not work on GPU
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.03),
            'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 10, 100),
            'depth': trial.suggest_int('depth', 2, 10),
            'bootstrap_type': trial.suggest_categorical(
                'bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']
            ),
        }

        if params['bootstrap_type'] == 'Bayesian':
            params['bagging_temperature'] = trial.suggest_float(
                'bagging_temperatur', 0, 10)
        elif params['bootstrap_typ'] == 'Bernoulli':
            params['subsample'] = trial.suggest_float('subsample', 0.1, 1)

        logging.info(f'\n----------------------\n{params}')

        return performance_scoring(
            model=CatBoostClassificationModel,
            features=features,
            target=target,
            peaks_indices=peaks_indices,
            model_params=params,
            k_folds=kfolds,
            metric_function=metric_function,
            weights=weights,
            categorical_features=categorical_features,
            random_state=random_state,
        )
