"""
Implementation of linear models for regression and classification.
See: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
"""
import warnings
import sklearn
from sklearn import linear_model
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
from tabular_ml.functions import (
    performance_scoring,
)
from tabular_ml.base import (
    MLModel,
    ModelTypes,
)
from tabular_ml.factory import ImplementedModel


@ImplementedModel
class LinearRegressionModel(MLModel):

    model_type: ModelTypes = 'regression'

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
        trained_model: sklearn.linear_model.LinearRegression,
        x_test: pd.DataFrame,
        categorical_features: Optional[List[str]] = None,
    ) -> np.ndarray:
        return trained_model.predict(x_test)

    @staticmethod
    def train_and_predict(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        model_params: Dict[str, Any],
        weights_train: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> Tuple[sklearn.linear_model.LinearRegression, np.ndarray]:
        # ordinarily encode categorical features?

        # train model
        model = LinearRegressionModel.train_model(
            x_train,
            y_train,
            model_params,
            weights_train,
            categorical_features,
        )

        return (
            model,
            LassoRegressionModel.make_predictions(
                trained_model=model,
                x_test=x_test,
            ),
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


@ImplementedModel
class RidgeRegressionModel(MLModel):

    model_type: ModelTypes = 'regression'

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
        trained_model: sklearn.linear_model.Ridge,
        x_test: pd.DataFrame,
        categorical_features: Optional[List[str]] = None,
    ) -> np.ndarray:
        return trained_model.predict(x_test)

    @staticmethod
    def train_and_predict(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        model_params: Dict[str, Any],
        weights_train: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> Tuple[sklearn.linear_model.Ridge, np.ndarray]:
        # one-hot-encode encode test categorical features

        # train model
        model = RidgeRegressionModel.train_model(
            x_train,
            y_train,
            model_params,
            weights_train,
            categorical_features,
        )

        return (
            model,
            RidgeRegressionModel.make_predictions(
                trained_model=model,
                x_test=x_test,
            ),
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
            model_class=RidgeRegressionModel,
            features=features,
            target=target,
            kfolds=kfolds,
            metric_function=metric_function,
            model_params=model_params,
            weights=weights,
            categorical_features=categorical_features,
            random_state=random_state,
        )


@ImplementedModel
class LassoRegressionModel(MLModel):

    model_type: ModelTypes = 'regression'

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
        trained_model: sklearn.linear_model.Lasso,
        x_test: pd.DataFrame,
        categorical_features: Optional[List[str]] = None,
    ) -> np.ndarray:
        return trained_model.predict(x_test)

    @staticmethod
    def train_and_predict(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        model_params: Dict[str, Any],
        weights_train: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> Tuple[sklearn.linear_model.Lasso, np.ndarray]:
        # one-hot-encode encode test categorical features

        # train model
        model = LassoRegressionModel.train_model(
            x_train,
            y_train,
            model_params,
            weights_train,
            categorical_features,
        )

        return (
            model,
            LassoRegressionModel.make_predictions(
                trained_model=model,
                x_test=x_test,
            ),
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
            model_class=LassoRegressionModel,
            features=features,
            target=target,
            kfolds=kfolds,
            metric_function=metric_function,
            model_params=model_params,
            weights=weights,
            categorical_features=categorical_features,
            random_state=random_state,
        )


@ImplementedModel
class ElasticNetRegressionModel(MLModel):

    model_type: ModelTypes = 'regression'

    @staticmethod
    def train_model(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        model_params: Dict[str, Any],
        weights_train: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> sklearn.linear_model.ElasticNet:
        # one-hot-encode encode test categorical features

        # set up training weights
        if weights_train is not None:
            weights_train = weights_train.values

        # init model
        model = sklearn.linear_model.ElasticNet(**model_params)

        # return the trained model
        return model.fit(
            x_train,
            y_train,
            sample_weight=weights_train,
        )

    @staticmethod
    def make_predictions(
        trained_model: sklearn.linear_model.ElasticNet,
        x_test: pd.DataFrame,
        categorical_features: Optional[List[str]] = None,
    ) -> np.ndarray:
        # TODO: consider adding a shared warning to check that the expected model
        # instance type is passed thru
        return trained_model.predict(x_test)

    @staticmethod
    def train_and_predict(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        model_params: Dict[str, Any],
        weights_train: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> Tuple[sklearn.linear_model.ElasticNet, np.ndarray]:
        # one-hot-encode encode test categorical features

        # train model
        model = ElasticNetRegressionModel.train_model(
            x_train,
            y_train,
            model_params,
            weights_train,
            categorical_features,
        )

        return (
            model,
            ElasticNetRegressionModel.make_predictions(
                trained_model=model,
                x_test=x_test,
            ),
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
            'l1_ratio': trial.suggest_uniform('l1_ratio', 0, 1),
            # 'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        }

        return performance_scoring(
            model_class=ElasticNetRegressionModel,
            features=features,
            target=target,
            kfolds=kfolds,
            metric_function=metric_function,
            model_params=model_params,
            weights=weights,
            categorical_features=categorical_features,
            random_state=random_state,
        )


@ImplementedModel
class BayesianRidgeRegressionModel(MLModel):

    model_type: ModelTypes = 'regression'

    @staticmethod
    def train_model(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        model_params: Dict[str, Any],
        weights_train: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> sklearn.linear_model.BayesianRidge:
        # one-hot-encode encode test categorical features

        # set up training weights
        if weights_train is not None:
            weights_train = weights_train.values

        # init model
        model = sklearn.linear_model.BayesianRidge(**model_params)

        # return the trained model
        return model.fit(
            x_train,
            y_train,
            sample_weight=weights_train,
        )

    @staticmethod
    def make_predictions(
        trained_model: sklearn.linear_model.BayesianRidge,
        x_test: pd.DataFrame,
        categorical_features: Optional[List[str]] = None,
    ) -> np.ndarray:
        return trained_model.predict(x_test)

    @staticmethod
    def train_and_predict(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        model_params: Dict[str, Any],
        weights_train: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> Tuple[sklearn.linear_model.BayesianRidge, np.ndarray]:
        # one-hot-encode encode test categorical features

        # train model
        model = BayesianRidgeRegressionModel.train_model(
            x_train,
            y_train,
            model_params,
            weights_train,
            categorical_features,
        )

        return (
            model,
            BayesianRidgeRegressionModel.make_predictions(
                trained_model=model,
                x_test=x_test,
            ),
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
            'n_iter': trial.suggest_int('n_iter', 200, 1000),
            'alpha_1': trial.suggest_loguniform('alpha_1', 1e-5, 1e5),
            'alpha_2': trial.suggest_loguniform('alpha_2', 1e-5, 1e5),
            'lambda_1': trial.suggest_loguniform('lambda_1', 1e-5, 1e5),
            'lambda_2': trial.suggest_loguniform('lambda_2', 1e-5, 1e5),
            # 'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        }

        return performance_scoring(
            model_class=BayesianRidgeRegressionModel,
            features=features,
            target=target,
            kfolds=kfolds,
            metric_function=metric_function,
            model_params=model_params,
            weights=weights,
            categorical_features=categorical_features,
            random_state=random_state,
        )
