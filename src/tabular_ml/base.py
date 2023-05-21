"""Base class for ML models"""
import abc
import warnings
import numpy as np
import pandas as pd
from optuna import Trial
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Literal,
)

ModelTypes = Literal['classification', 'regression']
OptunaRangeDict = Dict[str, Tuple[float, float] | Tuple[int, int] | List[str]]


class MLModel(abc.ABC):
    """Signature for a ML model"""

    model_type: ModelTypes
    optuna_param_ranges: OptunaRangeDict

    @abc.abstractclassmethod
    def train_model(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        model_params: Dict[str, Any],
        weights_train: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> object:
        raise NotImplementedError

    @abc.abstractstaticmethod
    def make_predictions(
        trained_model: object,
        x_test: pd.DataFrame,
        categorical_features: Optional[List[str]] = None,
    ) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractclassmethod
    def train_and_predict(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        model_params: Dict[str, Any],
        weights_train: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> Tuple[object, np.ndarray]:
        raise NotImplementedError

    @abc.abstractclassmethod
    def objective(
        trial: Trial,
        features: pd.DataFrame,
        target: pd.Series,
        kfolds: int,
        metric_function: callable,
        weights: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None,
        random_state: Optional[int] = None,
        custom_param_ranges: Optional[OptunaRangeDict] = None,
    ) -> float:
        raise NotImplementedError

    @classmethod
    def get_optuna_ranges(
        cls,
        custom_optuna_ranges: Optional[OptunaRangeDict] = None,
    ) -> OptunaRangeDict:
        """Get the optuna ranges for the model (and/or updates them)"""

        # return base ranges if no custom ranges
        if not custom_optuna_ranges:
            return cls.optuna_param_ranges

        # update the base ranges with the custom ranges if desired
        new_optuna_ranges = cls.optuna_param_ranges.copy()
        for key, value in custom_optuna_ranges.items():
            if key not in new_optuna_ranges:
                warnings.warn(
                    f'{key} is not a supported parameter! Skipping.',
                )
                continue
            if isinstance(value, type(new_optuna_ranges[key])):
                new_optuna_ranges[key] = value

            else:
                warnings.warn(
                    f'Custom optuna range for {key} is not the correct type='
                    f'{type(new_optuna_ranges[key])}. Using default range instead.',
                )
        return new_optuna_ranges
