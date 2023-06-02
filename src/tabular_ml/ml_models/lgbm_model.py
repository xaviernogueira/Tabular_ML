"""
LightGBM Regression.
"""
import lightgbm
import logging
import warnings
import pandas as pd
import numpy as np
from optuna.trial import Trial
from typing import (
    Dict,
    Literal,
    Tuple,
    Optional,
    Any,
    List,
    get_args,
)
from tabular_ml.functions import (
    performance_scoring,
)
from tabular_ml.utilities import (
    get_optuna_ranges,
    suggest_optuna_params,
)
from tabular_ml.base import (
    MLModel,
    ModelTypes,
    OptunaRangeDict,
)
from tabular_ml.factory import ModelFactory

# Define a custom callback to suppress evaluation logging


class SuppressEvalCallback:
    """Suppresses LightGBM evaluation logging."""

    def __init__(self):
        pass

    def __call__(self, env, **kwargs):
        pass


MultiClassMetrics: Literal = Literal[
    'multi_logloss',
    'multiclass',
    'softmax',
    'multiclassova',
    'multiclass_ova',
    'ova',
    'ovr',
    'multi_error',
    'auc_mu',
]


class BaseLightGBMModel(MLModel):

    model_type: ModelTypes = None
    optuna_param_ranges: OptunaRangeDict = None

    @classmethod
    def train_model(
        cls,
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

        # train model and return log loss score for testing
        callbacks: List[object] = [SuppressEvalCallback()]

        if 'early_stopping_rounds' in model_params.keys():
            early_stopping_rounds = model_params.pop('early_stopping_rounds')
            callbacks.append(lightgbm.early_stopping(early_stopping_rounds))
        else:
            early_stopping_rounds = None
        if 'num_boost_round' in model_params.keys():
            num_boost_round = model_params.pop('num_boost_round')
        else:
            num_boost_round = 100

        # suppress lightgbm logging bv default
        if 'verbose' not in model_params.keys():
            model_params['verbose'] = -1

        # set up classification objective and eval metric
        if cls.model_type == 'classification':
            model_params = cls._objective_and_eval(
                model_params,
                y_train,
            )

        # return the trained model
        return lightgbm.train(
            params=model_params,
            train_set=train_data_ds,
            callbacks=callbacks,
            early_stopping_rounds=early_stopping_rounds,
            valid_sets=[train_data_ds],
            num_boost_round=num_boost_round,
        )

    @ staticmethod
    def make_predictions(
        trained_model: lightgbm.Booster,
        x_test: pd.DataFrame,
        categorical_features: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Makes predictions with LightGBM"""

        return trained_model.predict(x_test)

    @ classmethod
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

    @ classmethod
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
        LightGBM parameter search space for optuna.
        """
        # get parameter ranges
        param_ranges = get_optuna_ranges(
            cls.optuna_param_ranges,
            custom_optuna_ranges=custom_optuna_ranges,
        )

        # set up parameters
        function_mapping = {
            'objective': trial.suggest_categorical,
            'metric': trial.suggest_categorical,
            'early_stopping_rounds': trial.suggest_int,
            'num_leaves': trial.suggest_int,
            'lambda_l1': trial.suggest_float,
            'lambda_l2': trial.suggest_float,
            'learning_rate': trial.suggest_float,
            'max_depth': trial.suggest_int,
            'colsample_bytree': trial.suggest_float,
            'min_child_weight': trial.suggest_float,
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


@ ModelFactory.implemented_model
class LightGBMRegressionModel(BaseLightGBMModel):
    model_type: ModelTypes = 'regression'
    optuna_param_ranges: OptunaRangeDict = {
        'objective': ['regression'],
        'metric': ['mae', 'rmse'],
        'early_stopping_rounds': (10, 100),
        'num_leaves': (20, 150),
        'lambda_l1': (0.0, 0.5),
        'lambda_l2': (0.0, 0.5),
        'learning_rate': (0.001, 0.4),
        'max_depth': (0, 8),
        'colsample_bytree': (0.5, 1.0),
        'min_child_weight': (0.1, 10),
        'num_boost_round': (250, 1500),
    }


@ ModelFactory.implemented_model
class LightGBMClassificationModel(BaseLightGBMModel):
    model_type: ModelTypes = 'classification'
    optuna_param_ranges: OptunaRangeDict = {
        'early_stopping_rounds': (10, 100),
        'num_leaves': (20, 150),
        'lambda_l1': (0.0, 0.5),
        'lambda_l2': (0.0, 0.5),
        'learning_rate': (0.001, 0.4),
        'max_depth': (0, 8),
        'colsample_bytree': (0.5, 1.0),
        'min_child_weight': (0.1, 10),
        'num_boost_round': (250, 1500),
    }

    @ staticmethod
    def _objective_and_eval(
        model_params: Dict[str, Any],
        y_train: pd.Series,
    ) -> Dict[str, Any]:
        """Sets the objective and eval metric for classification."""

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
                model_params['objective'] = 'binary'
            if not 'eval_metric' in model_params.keys():
                model_params['eval_metric'] = 'binary_logloss'
            del model_params['num_class']

        else:
            if not 'objective' in model_params.keys():
                model_params['objective'] = 'softmax'
            elif 'multi' not in model_params['objective']:
                raise ValueError(
                    'Must choose a multi-class classification suitable objective function.',
                )

            if not 'metric' in model_params.keys():
                model_params['metric'] = 'multi_logloss'
            elif model_params['metric'] not in get_args(MultiClassMetrics):
                raise ValueError(
                    f'Must choose one of the following multi-class classification '
                    f'suitable eval metrics: {MultiClassMetrics}.',
                )
        return model_params
