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
from tabular_ml.utilities import get_optuna_ranges
from tabular_ml.factory import ModelFactory

LinearModels = Union[
    sklearn.linear_model.LinearRegression,
    sklearn.linear_model.Ridge,
    sklearn.linear_model.Lasso,
    sklearn.linear_model.ElasticNet,
    sklearn.linear_model.BayesianRidge,
]


class BaseLinearModel(MLModel):
    """Base class for linear models.

    NOTE: This is not intended to be used directly!
    """

    model_type: ModelTypes = None
    model_object: LinearModels = None
    optuna_param_ranges = None

    @classmethod
    def train_model(
        cls,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        model_params: Dict[str, Any],
        weights_train: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> object:

        # set up training weights
        if weights_train is not None:
            weights_train = weights_train.values

        # init model
        model = cls.model_object(**model_params)

        # return the trained model
        return model.fit(
            x_train,
            y_train,
            sample_weight=weights_train,
        )

    @staticmethod
    def make_predictions(
        trained_model: object,
        x_test: pd.DataFrame,
        categorical_features: Optional[List[str]] = None,
    ) -> np.ndarray:
        return trained_model.predict(x_test)

    @classmethod
    def train_and_predict(
        cls,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        model_params: Dict[str, Any],
        weights_train: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> Tuple[object, np.ndarray]:
        # ordinarily encode categorical features?

        # train model
        model = cls.train_model(
            x_train,
            y_train,
            model_params,
            weights_train,
            categorical_features,
        )

        return (
            model,
            cls.make_predictions(
                trained_model=model,
                x_test=x_test,
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

        warnings.warn(
            f'There are no hyperparameters to tune for {type(cls.model_object)}',
        )
        raise NotImplementedError


@ModelFactory.implemented_model
class LinearRegressionModel(BaseLinearModel):

    model_type: ModelTypes = 'regression'
    model_object: LinearModels = sklearn.linear_model.LinearRegression
    optuna_param_ranges = None


@ModelFactory.implemented_model
class RidgeRegressionModel(BaseLinearModel):

    model_type: ModelTypes = 'regression'
    model_object: LinearModels = sklearn.linear_model.Ridge
    optuna_param_ranges: OptunaRangeDict = {
        'alpha': (1e-5, 1e5),
    }

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

        # get optuna params
        param_ranges = get_optuna_ranges(
            cls.optuna_param_ranges,
            custom_optuna_ranges=custom_optuna_ranges,
        )

        # set up training weights
        if weights is not None:
            weights = weights.values

        # set up model params
        model_params = {
            'alpha': trial.suggest_float(
                'alpha',
                param_ranges['alpha'][0],
                param_ranges['alpha'][-1],
            ),
        }

        return performance_scoring(
            model=cls,
            features=features,
            target=target,
            k_folds=kfolds,
            metric_function=metric_function,
            model_params=model_params,
            weights=weights,
            categorical_features=categorical_features,
            random_state=random_state,
        )


@ModelFactory.implemented_model
class LassoRegressionModel(BaseLinearModel):

    model_type: ModelTypes = 'regression'
    model_object: LinearModels = sklearn.linear_model.Lasso
    optuna_param_ranges: OptunaRangeDict = {
        'alpha': (1e-5, 1e5),
    }

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

        # get optuna params
        param_ranges = get_optuna_ranges(
            cls.optuna_param_ranges,
            custom_optuna_ranges=custom_optuna_ranges,
        )

        # set up model params
        model_params = {
            'alpha': trial.suggest_float(
                'alpha',
                param_ranges['alpha'][0],
                param_ranges['alpha'][-1],
            ),
        }

        return performance_scoring(
            model=cls,
            features=features,
            target=target,
            k_folds=kfolds,
            metric_function=metric_function,
            model_params=model_params,
            weights=weights,
            categorical_features=categorical_features,
            random_state=random_state,
        )


@ModelFactory.implemented_model
class ElasticNetRegressionModel(BaseLinearModel):

    model_type: ModelTypes = 'regression'
    model_object: LinearModels = sklearn.linear_model.ElasticNet
    optuna_param_ranges: OptunaRangeDict = {
        'alpha': (1e-5, 1e5),
        'l1_ratio': (0, 1),
    }

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

        # get optuna params
        param_ranges = get_optuna_ranges(
            cls.optuna_param_ranges,
            custom_optuna_ranges=custom_optuna_ranges,
        )

        function_mapping = {
            'alpha': trial.suggest_float,
            'l1_ratio': trial.suggest_float,
        }

        # set up model params
        model_params = {}
        for param in param_ranges.keys():
            model_params[param] = function_mapping[param](
                param,
                param_ranges[param][0],
                param_ranges[param][-1],
            )

        return performance_scoring(
            model=cls,
            features=features,
            target=target,
            k_folds=kfolds,
            metric_function=metric_function,
            model_params=model_params,
            weights=weights,
            categorical_features=categorical_features,
            random_state=random_state,
        )


@ModelFactory.implemented_model
class BayesianRidgeRegressionModel(BaseLinearModel):

    model_type: ModelTypes = 'regression'
    model_object: LinearModels = sklearn.linear_model.BayesianRidge
    optuna_param_ranges: OptunaRangeDict = {
        'n_iter': (200, 1000),
        'alpha_1': (1e-5, 1e5),
        'alpha_2': (1e-5, 1e5),
        'lambda_1': (1e-5, 1e5),
        'lambda_2': (1e-5, 1e5),
    }

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

        # get optuna params
        param_ranges = get_optuna_ranges(
            cls.optuna_param_ranges,
            custom_optuna_ranges=custom_optuna_ranges,
        )

        function_map = {
            'n_iter': trial.suggest_int,
            'alpha_1': trial.suggest_float,
            'alpha_2': trial.suggest_float,
            'lambda_1': trial.suggest_float,
            'lambda_2': trial.suggest_float,
        }

        # set up model params
        model_params = {}
        for param in param_ranges.keys():
            model_params[param] = function_map[param](
                param,
                param_ranges[param][0],
                param_ranges[param][-1],
            )

        return performance_scoring(
            model=cls,
            features=features,
            target=target,
            k_folds=kfolds,
            metric_function=metric_function,
            model_params=model_params,
            weights=weights,
            categorical_features=categorical_features,
            random_state=random_state,
        )
