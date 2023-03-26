"""
Implementation of linear models for regression and classification.
See: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
"""
import sklearn
import numpy as np
import pandas as pd
from optuna.trial import Trial
from typing import (
    Dict,
    List,
    Tuple,
    Optional,
    Any,
)
from ml_model_shared import (
    MLModel,
    performance_scoring,
)


class LinearRegressor(MLModel):

    @staticmethod
    def train_model(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        model_params: Dict[str, Any],
        weights_train: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> sklearn.linear_model.LinearRegression:

        # set up training weights
        if weights_train is not None:
            weights_train = weights_train.values

        # init model
        model = sklearn.linear_model.LinearRegression(**model_params)

        # return the trained model
        return model.fit(
            x_train,
            y_train,
            sample_weight=weights_train,
        )

    @staticmethod
    def make_predictions(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        model_params: Dict[str, Any],
        weights_train: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> Tuple[sklearn.linear_model.LinearRegression, np.ndarray]:
        # ordinally encode categorical features?

        # train model
        model = LinearRegressor.train_model(
            x_train,
            y_train,
            model_params,
            weights_train,
            categorical_features,
        )

        return (
            model,
            model.predict(x_test),
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

        print('There are no hyperparameters to tune for LinearRegression.')
        raise NotImplementedError


class RidgeRegression(MLModel):

    @staticmethod
    def train_model(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        model_params: Dict[str, Any],
        weights_train: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> sklearn.linear_model.Ridge:
        # one-hot-encode encode test categorical features

        # set up training weights
        if weights_train is not None:
            weights_train = weights_train.values

        # init model
        model = sklearn.linear_model.Ridge(**model_params)

        # return the trained model
        return model.fit(
            x_train,
            y_train,
            sample_weight=weights_train,
        )

    @staticmethod
    def make_predictions(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        model_params: Dict[str, Any],
        weights_train: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> Tuple[sklearn.linear_model.Ridge, np.ndarray]:
        # one-hot-encode encode test categorical features

        # train model
        model = RidgeRegression.train_model(
            x_train,
            y_train,
            model_params,
            weights_train,
            categorical_features,
        )

        return (
            model,
            model.predict(x_test),
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

        # set up training weights
        if weights is not None:
            weights = weights.values

        # set up model params
        model_params = {
            'alpha': trial.suggest_loguniform('alpha', 1e-5, 1e5),
            # 'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        }

        return performance_scoring(
            model_class=RidgeRegression,
            features=features,
            target=target,
            kfolds=kfolds,
            metric_function=metric_function,
            model_params=model_params,
            weights=weights,
            categorical_features=categorical_features,
            random_state=random_state,
        )


class LassoRegressor(MLModel):

    @staticmethod
    def train_model(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        model_params: Dict[str, Any],
        weights_train: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> sklearn.linear_model.Lasso:
        # one-hot-encode encode test categorical features

        # set up training weights
        if weights_train is not None:
            weights_train = weights_train.values

        # init model
        model = sklearn.linear_model.Lasso(**model_params)

        # return the trained model
        return model.fit(
            x_train,
            y_train,
            sample_weight=weights_train,
        )

    @staticmethod
    def make_predictions(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        model_params: Dict[str, Any],
        weights_train: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> Tuple[sklearn.linear_model.Lasso, np.ndarray]:
        # one-hot-encode encode test categorical features

        # train model
        model = LassoRegression.train_model(
            x_train,
            y_train,
            model_params,
            weights_train,
            categorical_features,
        )

        return (
            model,
            model.predict(x_test),
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

        # set up model params
        model_params = {
            'alpha': trial.suggest_loguniform('alpha', 1e-5, 1e5),
            # 'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        }

        return performance_scoring(
            model_class=LassoRegressor,
            features=features,
            target=target,
            kfolds=kfolds,
            metric_function=metric_function,
            model_params=model_params,
            weights=weights,
            categorical_features=categorical_features,
            random_state=random_state,
        )
