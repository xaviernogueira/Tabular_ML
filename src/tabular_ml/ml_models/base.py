"""Base class for ML models"""
import abc
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


class MLModel(abc.ABC):
    """Signature for a ML model"""

    model_type: ModelTypes

    @abc.abstractclassmethod
    def train_model(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        model_params: Dict[str, Any],
        weights_train: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> object:
        raise NotImplementedError

    @abc.abstractclassmethod
    def make_predictions(
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
    ) -> float:
        raise NotImplementedError
