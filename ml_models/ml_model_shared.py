"""Stores shared MLModel ABC, as well as functions for all MLModel implementations"""
import dataclasses
import time
import logging
import numpy as np
import pandas as pd
import sklearn.metrics
import sklearn.model_selection
import optuna
import abc
from datetime import datetime
from optuna.trial import Trial
from typing import (
    Union,
    List,
    Dict,
    Tuple,
    Optional,
    Any,
)


class MLModel(abc.ABC):
    """Signature for a ML model"""

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


@dataclasses.dataclass
class KFoldOutput:
    """A class to store Regression KFold CV outputs"""
    n_splits: int
    random_state: Union[int, None]
    metric_function: str
    metric_function_kwargs: Dict[str, Any]
    using_training_weights: bool
    model_names: List[str]
    model_params: Dict[str, Dict[str, Any]]  # model names as keys
    raw_model_scores: Dict[str, float]  # model names as keys
    adj_model_scores: Dict[str, float]  # model names as keys
    model_test_losses: Dict[str, List[float]]  # model names as keys
    # k fold index as keys
    model_objects_by_fold = Dict[str, Dict[str, object]]
    ensemble_raw_score: float
    ensemble_adj_score: float
    run_time: int


def get_ensemble_prediction(
    model_predictions_dict: Dict[str, np.ndarray],
) -> float:
    """Combines numpy array predictions"""
    return np.mean(
        list(model_predictions_dict.values()),
        axis=0,
    )


def k_fold_cv(
    x_data: pd.DataFrame,
    y_data: pd.Series,
    model_classes: Dict[str, MLModel],
    model_params: Dict[str, Dict],
    weights_data: Optional[pd.Series] = None,
    categorical_features: Optional[List[str]] = None,
    n_splits: Optional[int] = None,
    metric_function: Optional[callable] = None,
    metric_function_kwargs: Optional[Dict[str, Any]] = None,
    random_state: Optional[int] = None,
) -> KFoldOutput:
    """Runs a Regression K-fold CV for any set of models.

    Arguments:
        prediction_functions: a dict with model names as keys,
            and model prediction functions as values (matching args).
        model_params: a dict with model names as keys, 
            and parameters for that model as values.
        metric_function: A function with the signature 
            f(y_true, y_preds, **kwargs) -> float. Default is R-Squared.
        metric_function_kwargs: Kwargs to pass to the metric function.
    Returns:
        A populated KFoldOutput dataclass with relevant info.

    NOTE: as of now this only works with Logloss -> generalize later?
    """
    # start timer
    p1 = time.perf_counter()

    # init logging
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler('k_fold_CV_logging.log'),
            logging.StreamHandler(),
        ],
    )

    # init output lists
    model_test_losses = {}
    raw_model_scores = {}
    adj_model_scores = {}
    model_objects_by_fold = {}

    model_names = list(model_classes.keys())

    for name in model_names:
        # store model losses for each fold
        model_test_losses[name] = []

        # use defaults if params not specified
        if name not in model_params.keys():
            model_params[name] = {}
            logging.info(f'Using default params for {name}')
    model_test_losses['ensemble'] = []

    # initiate scoring metrics
    if metric_function is None:
        metric_function = sklearn.metrics.r2_score
    if metric_function_kwargs is None:
        metric_function_kwargs = {}

    # run k-folds
    if n_splits is None:
        n_splits = 5
    kfolds = sklearn.model_selection.KFold(
        n_splits=n_splits,
        random_state=random_state,
        shuffle=True,
    )

    # run the K-fold CV
    for i, (train_idxs, test_idxs) in enumerate(kfolds.split(x_data)):

        x_train, x_test = x_data.loc[train_idxs], x_data.loc[test_idxs]
        y_train, y_test = y_data.loc[train_idxs], y_data.loc[test_idxs]

        # add training weights if necessary
        if isinstance(weights_data, pd.Series):
            weights_train = weights_data[train_idxs]
        else:
            weights_train = None

        # get predictions for each model, calculate score
        model_predictions = {}
        model_objects = {}
        for model_name, model_class in model_classes.items():
            logging.info(f'Split {i} | {model_name} - {datetime.now()}')
            params = model_params[model_name].copy()

            model_objects[model_name], model_predictions[model_name] = model_class.make_predictions(
                x_train,
                y_train,
                x_test,
                model_params=params,
                weights_train=weights_train,
                categorical_features=categorical_features,
            )

            model_test_losses[model_name].append(
                metric_function(
                    y_test,
                    model_predictions[model_name],
                    **metric_function_kwargs,
                ),
            )
        # store model objects from the K-fold
        model_objects_by_fold[i] = model_objects

        # get ensemble prediction, calculate score
        ensemble_preds = get_ensemble_prediction(
            model_predictions_dict=model_predictions,
        )
        model_test_losses['ensemble'].append(
            metric_function(
                y_test,
                ensemble_preds,
                **metric_function_kwargs,
            )
        )

    # get final outputs
    for model_name in model_test_losses.keys():
        test_losses = model_test_losses[model_name]
        raw_model_scores[model_name] = np.mean(np.array(test_losses))
        adj_model_scores[model_name] = (
            np.mean(np.array(test_losses)) + np.std(np.array(test_losses))
        )

    # stop timer
    p2 = time.perf_counter()

    # return output as a class
    out_class = KFoldOutput(
        n_splits=n_splits,
        random_state=random_state,
        metric_function=metric_function.__name__,
        metric_function_kwargs=metric_function_kwargs,
        using_training_weights=isinstance(weights_data, pd.Series),
        model_names=model_names,
        model_params=model_params,
        raw_model_scores=raw_model_scores,
        adj_model_scores=adj_model_scores,
        model_test_losses=model_test_losses,
        model_objects_by_fold=model_objects_by_fold,
        ensemble_raw_score=raw_model_scores['ensemble'],
        ensemble_adj_score=adj_model_scores['ensemble'],
        run_time=p2-p1,
    )
    logging.info(f'Done! - {datetime.now()}')
    return out_class


def performance_scoring(
    model: MLModel,
    features: pd.DataFrame,
    target: pd.Series,
    model_params: Dict[str, Union[float, int, str]],
    k_folds: int,
    metric_function: callable,
    weights: Optional[pd.Series] = None,
    categorical_features: Optional[List[str]] = None,
    random_state: Optional[int] = None,
) -> float:
    """
    K-fold CV wrapper for Optuna optimization
    """
    # run K-fold CV and get scores
    kfolds_output = k_fold_cv(
        x_data=features,
        y_data=target,
        model_classes={model.__name__: model},
        model_params={model.__name__: model_params},
        weights_data=weights,
        categorical_features=categorical_features,
        n_splits=k_folds,
        metric_function=metric_function,
        random_state=random_state,
    )
    adj_score = kfolds_output.adj_model_scores[model.__name__]
    logging.info(f'K-fold CV output: {dataclasses.asdict(kfolds_output)}')

    return adj_score


def find_optimal_parameters(
    model: MLModel,
    features: pd.DataFrame,
    target: pd.Series,
    metric_function: callable,
    direction: Optional[str] = None,
    n_trials: int = 20,
    timeout: Optional[int] = None,
    kfolds: int = 5,
    weights: Optional[pd.Series] = None,
    categorical_features: Optional[List[str]] = None,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Runs optuna optimization for a MLModel"""

    # init logging
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(f'ml_model_optimization.log'),
            logging.StreamHandler(),
        ],
    )

    # create the optuna study
    logging.info(
        f'Starting {model.__name__} optimization w/ n_trials={n_trials}, and timeout={timeout}'
    )

    logging.info(
        f'Using metric function: {metric_function.__name__}'
    )
    # record time
    start_time = datetime.now()

    # workaround to pass argument into the objective https://www.kaggle.com/general/261870
    def objective_func(trial): return model.objective(
        trial,
        features,
        target,
        kfolds=kfolds,
        metric_function=metric_function,
        weights=weights,
        categorical_features=categorical_features,
        random_state=random_state,
    )

    # assume metric direction when possible
    direction_dict = {
        'mean_absolute_error': 'minimize',
        'log_loss': 'minimize',
        'r2_score': 'maximize',
    }
    if str(metric_function.__name__) in direction_dict.keys():
        direction = direction_dict[str(metric_function.__name__)]
    else:
        logging.warn(
            f'Optimal direction cannot be inferred from metric_function: '
            f'{metric_function.__name__}. Please set param:direction to '
            f'maximize or minimize. Default is minimize!'
        )
        direction = 'minimize'

    # run the study
    study = optuna.create_study(direction=direction)
    study.optimize(
        objective_func,
        n_trials,
        timeout,
    )

    # record end time and key results
    end_time = datetime.now()
    logging.info(
        f'{metric_function.__name__} Optimization complete! Took {end_time - start_time}.\n'
    )

    # log key stats
    best_trial = study.best_trial
    logging.info(f'Number of finished trials: {len(study.trials)}')

    logging.info(f'Best trial = {best_trial}')
    logging.info(f'Best trial value = {best_trial.value}\n')

    logging.info(f'Best hyperparameters:')

    for key, value in best_trial.params.items():
        logging.info(f'{key} = {value}')
    logging.info('-------------------------------------------\n\n\n')

    # return best params
    return best_trial.params
