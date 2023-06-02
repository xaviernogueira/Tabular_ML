"""
XGBoost Regression.
"""
import xgboost
import logging
import warnings
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
from tabular_ml.utilities import (
    get_optuna_ranges,
    suggest_optuna_params,
)
from tabular_ml.factory import ModelFactory


class BaseXGBoostModel(MLModel):
    """A class for shared XGBoost methdods.

    This class is not intended to be used directly.
    Note that currently only XGBoost booster=gbtree is supported.
    """
    model_type = None

    @classmethod
    def train_model(
        cls,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        model_params: Dict[str, Any],
        weights_train: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> xgboost.Booster:

        # set up training weights
        if weights_train is not None:
            weights_train = weights_train.values

        # deal with training categorical features
        if categorical_features is not None:
            for cat_feat in categorical_features:
                x_train[cat_feat] = x_train[cat_feat].astype('category')

        train_data_matrix = xgboost.DMatrix(
            data=x_train,
            label=y_train,
            weight=weights_train,
            nthread=-1,  # max multithreading
            enable_categorical=bool(categorical_features),
        )

        evals = [
            (train_data_matrix, 'train'),
        ]

        # train model and return log loss score for testing
        if 'early_stopping_rounds' in model_params.keys():
            early_stopping_rounds = model_params.pop('early_stopping_rounds')
        else:
            early_stopping_rounds = None
        if 'num_boost_round' in model_params.keys():
            num_boost_round = model_params.pop('num_boost_round')
        else:
            # XGBoost requires an int, this is the libraries default
            num_boost_round = 10
        if 'verbose_eval' in model_params.keys():
            if isinstance(model_params['verbose_eval'], bool):
                verbose_eval = model_params.pop('verbose_eval')
        else:
            verbose_eval = False

        # set up classification objective and eval metric
        if cls.model_type == 'classification':
            model_params = cls._objective_and_eval(
                model_params,
                y_train,
            )

        # return the trained model
        return xgboost.train(
            params=model_params,
            dtrain=train_data_matrix,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            num_boost_round=num_boost_round,
            verbose_eval=verbose_eval,
        )

    @classmethod
    def make_predictions(
        cls,
        trained_model: xgboost.Booster,
        x_test: pd.DataFrame,
        categorical_features: Optional[List[str]] = None,
    ) -> np.ndarray:
        # deal with testing categorical features
        if categorical_features is not None:
            for cat_feat in categorical_features:
                x_test[cat_feat] = x_test[cat_feat].astype('category')

        # load in testing data
        test_data_matrix = xgboost.DMatrix(
            data=x_test,
            nthread=-1,  # max multithreading
            enable_categorical=bool(categorical_features),
        )

        # control predicting classification probabilities
        if cls.model_type == 'classification':
            output_margin = True
        elif cls.model_type == 'regression':
            output_margin = False

        return trained_model.predict(
            test_data_matrix,
            output_margin=output_margin,
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
    ) -> Tuple[xgboost.Booster, np.ndarray]:

        # train the model
        xgb_model = cls.train_model(
            x_train,
            y_train,
            model_params.copy(),
            weights_train=weights_train,
            categorical_features=categorical_features,
        )

        # return prediction array
        return (
            xgb_model,
            cls.make_predictions(
                trained_model=xgb_model,
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
        XGBoost parameter search space for optuna.
        """

        # get parameter ranges
        param_ranges = get_optuna_ranges(
            cls.optuna_param_ranges,
            custom_optuna_ranges=custom_optuna_ranges,
        )

        # set up parameters
        function_mapping = {
            'objective': trial.suggest_categorical,
            'eval_metric': trial.suggest_categorical,
            'early_stopping_rounds': trial.suggest_int,
            'lambda': trial.suggest_int,
            'learning_rate': trial.suggest_float,
            'max_depth': trial.suggest_int,
            'colsample_bytree': trial.suggest_float,
            'num_boost_round': trial.suggest_int,
        }
        params = suggest_optuna_params(
            trial,
            param_ranges,
            function_mapping,
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


@ModelFactory.implemented_model
class XGBoostRegressionModel(BaseXGBoostModel):

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


@ModelFactory.implemented_model
class XGBoostClassificationModel(BaseXGBoostModel):

    model_type: ModelTypes = 'classification'
    optuna_param_ranges: OptunaRangeDict = {
        'early_stopping_rounds': (10, 100),
        'lambda': (3, 8),
        'learning_rate': (0.01, 0.4),
        'max_depth': (2, 8),
        'colsample_bytree': (0.5, 1.0),
        'num_boost_round': (250, 1500),
    }

    @staticmethod
    def _objective_and_eval(
        model_params: Dict[str, Any],
        y_train: pd.Series,
    ) -> Dict[str, Any]:
        """Set objective and eval metric for classification."""

        # identify whether we are doing binary or multi-class classification
        num_classes = len(np.unique(y_train))
        if not 'num_class' in model_params.keys():
            model_params['num_class'] = num_classes
        elif model_params['num_class'] != num_classes:
            warnings.warn(
                'Manually entered num_class does not match the number of classes in the data slice. '
                'This may be OK due to stochastic data slice, but is likely an error.',
            )

        # set up appropriate objective and eval metric
        if model_params['num_class'] <= 2:
            if not 'objective' in model_params.keys():
                model_params['objective'] = 'binary:logistic'
            if not 'eval_metric' in model_params.keys():
                model_params['eval_metric'] = 'logloss'
            del model_params['num_class']

        else:
            if not 'objective' in model_params.keys():
                model_params['objective'] = 'multi:softprob'
            elif 'multi' not in model_params['objective']:
                raise ValueError(
                    'Must choose a multi-class classification suitable objective function.',
                )

            if not 'eval_metric' in model_params.keys():
                model_params['eval_metric'] = 'mlogloss'
            elif model_params['eval_metric'][0] != 'm':
                raise ValueError(
                    'Must choose a multi-class classification suitable eval metric.',
                )
        return model_params
