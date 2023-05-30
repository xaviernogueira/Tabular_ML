"""Stores shared MLModel ABC, as well as functions for all MLModel implementations"""
import dataclasses
import time
import logging
import numpy as np
import pandas as pd
import sklearn.metrics
import sklearn.model_selection
import optuna
from datetime import datetime
from tabular_ml.base import (
    MLModel,
    KFoldOutput,
    OptunaRangeDict,
)
from tabular_ml.factory import ModelFactory
import tabular_ml.utilities as utilities
from pathlib import Path
from typing import (
    List,
    Dict,
    Optional,
    Any,
)


def k_fold_cv(
    x_data: pd.DataFrame,
    y_data: pd.Series,
    model_names: List[str],
    model_params: Dict[str, Dict[str, float | int | str]],
    weights_data: Optional[pd.Series] = None,
    categorical_features: Optional[List[str]] = None,
    n_splits: Optional[int] = None,
    metric_function: Optional[callable] = None,
    metric_function_kwargs: Optional[Dict[str, Any]] = None,
    random_state: Optional[int] = None,
    logging_file_path: Optional[str | Path] = None,
) -> KFoldOutput:
    """Runs K-fold CV for any set of models.

    Arguments:
        x_data: A pandas DataFrame of features.
        y_data: A pandas Series of targets.
        model_names: A list of valid model names.
            Valid names can be queried using ModelFactory.get_all_models().
        model_params: a dict with model names as keys,
            and parameters for that model as values.
        weights_data: A pandas Series of training weights.
        categorical_features: A list of categorical feature names.
        n_splits: # of K-Folds. Default is 5.
        metric_function: A function with the signature
            f(y_true, y_preds, **kwargs) -> float. Default is R-Squared.
        metric_function_kwargs: Kwargs to pass to the metric function.
        random_state: Random state to use for K-Folds.
        logging_file_path: Path to a log file.

    Returns:
        A populated KFoldOutput dataclass with relevant info.
    """
    # start timer
    p1 = time.perf_counter()

    # init logging
    handlers = [
        logging.StreamHandler(),
    ]
    if logging_file_path is not None:
        handlers.append(logging.FileHandler(logging_file_path))
    logging.basicConfig(
        level=logging.INFO,
        handlers=handlers,
    )

    # init output lists
    model_test_losses = {}
    raw_model_scores = {}
    adj_model_scores = {}
    model_objects_by_fold = {}

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
        for model_name in model_names:
            model_class = ModelFactory.get_model(model_name)

            logging.info(f'Split {i} | {model_name} - {datetime.now()}')
            params = model_params[model_name].copy()

            model_objects[model_name], model_predictions[model_name] = model_class.train_and_predict(
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
        ensemble_preds = utilities.get_ensemble_prediction(
            model_predictions_dict=model_predictions,
        )
        model_test_losses['ensemble'].append(
            metric_function(
                y_test,
                ensemble_preds,
                **metric_function_kwargs,
            ),
        )

    # get final outputs
    for model_name in model_test_losses.keys():
        test_losses = model_test_losses[model_name]
        raw_model_scores[model_name] = np.mean(np.array(test_losses))
        adj_model_scores[model_name] = utilities.get_adjusted_score(
            test_losses,
            metric_function,
        )

    # stop timer
    p2 = time.perf_counter()

    # return output as a class
    out_class = KFoldOutput(
        n_splits=n_splits,
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
        random_state=random_state,
    )
    logging.info(f'Done! - {datetime.now()}')
    return out_class


def performance_scoring(
    model: MLModel | str,
    features: pd.DataFrame,
    target: pd.Series,
    model_params: Dict[str, float | int | str],
    k_folds: int,
    metric_function: callable,
    weights: Optional[pd.Series] = None,
    categorical_features: Optional[List[str]] = None,
    random_state: Optional[int] = None,
) -> float:
    """
    K-fold CV wrapper for Optuna optimization
    """
    # get model class
    if isinstance(model, str):
        model = ModelFactory.get_model(model)

    # run K-fold CV and get scores
    kfolds_output = k_fold_cv(
        x_data=features,
        y_data=target,
        model_names={model.__name__: model},
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
    model: str,
    features: pd.DataFrame,
    target: pd.Series,
    metric_function: callable,
    direction: Optional[str] = None,
    n_trials: Optional[int] = None,
    timeout: Optional[int] = None,
    kfolds: Optional[int] = None,
    weights: Optional[pd.Series] = None,
    categorical_features: Optional[List[str]] = None,
    random_state: Optional[int] = None,
    custom_optuna_ranges: Optional[OptunaRangeDict] = None,
    logging_file_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Runs optuna optimization for a MLModel.

    Arguments:
        model: A valid model name.
            Valid names can be queried using ModelFactory.get_all_models().
        features: A pandas DataFrame of features.
        target: A pandas Series of targets.
        metric_function: A function with the signature
            f(y_true, y_preds, **kwargs) -> float.
        direction: 'maximize' or 'minimize' the metric function.
            If None, will infer direction from metric function.
        n_trials: # of trials to run. Default is 20.
        timeout: # of seconds to run.
        kfolds: # of K-Folds. Default is 5.
        weights: A pandas Series of training weights.
        categorical_features: A list of categorical feature names.
        random_state: Random state to use for K-Folds.
        custom_optuna_ranges: A dict of custom optuna ranges.
        logging_file_path: Path to a log file.

    Returns:
        A dict of optimal parameters for the model.
    """
    # set defaults
    if n_trials is None:
        n_trials = 20
    if kfolds is None:
        kfolds = 5

    # get model class
    model = ModelFactory.get_model(model)

    # init logging
    handlers = [
        logging.StreamHandler(),
    ]
    if logging_file_path is not None:
        handlers.append(logging.FileHandler(logging_file_path))
    logging.basicConfig(
        level=logging.INFO,
        handlers=handlers,
    )

    # create the optuna study
    logging.info(
        f'Starting {model.__name__} optimization w/ n_trials={n_trials}, '
        f'and timeout={timeout}',
    )

    logging.info(
        f'Using metric function: {metric_function.__name__}',
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
        custom_optuna_ranges=custom_optuna_ranges,
    )

    # assume metric direction when possible
    direction = utilities.get_metric_direction(metric_function)

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
        f'{metric_function.__name__} Optimization complete! '
        f'Took {end_time - start_time}.\n',
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
