"""Base class for ML models and functions"""
import abc
import numpy as np
import pandas as pd
from optuna import Trial
from dataclasses import dataclass
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
    optuna_param_ranges: OptunaRangeDict | None

    @abc.abstractstaticmethod
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


@dataclass
class KFoldOutput:
    """A class to store Regression KFold CV outputs.

    Attributes:
        n_splits: # of K-Folds
        random_state: random state used (can be None)
        metric_function: name of the metric function used
        metric_function_kwargs: kwargs passed to the metric function
        using_training_weights: whether or not training weights were used
        model_names: list of model names used
        model_params: dict of model names as keys, model params as values
        raw_model_scores: dict of model names as keys, raw scores as values
        adj_model_scores: dict of model names as keys, adj scores as values
        model_test_losses: dict of model names as keys, test losses as values
        model_objects_by_fold: dict of k fold index keys, model objects as values
        ensemble_raw_score: ensemble raw score
        ensemble_adj_score: ensemble adj score
        run_time: run time in seconds
    """
    n_splits: int
    metric_function: str
    metric_function_kwargs: Dict[str, Any]
    using_training_weights: bool
    model_names: List[str]
    model_params: Dict[str, Dict[str, float | int | str]]
    raw_model_scores: Dict[str, float]
    adj_model_scores: Dict[str, float]
    model_test_losses: Dict[str, List[float]]
    model_objects_by_fold: Dict[str, Dict[str, object]]
    ensemble_raw_score: float
    ensemble_adj_score: float
    run_time: float
    random_state: int | None
