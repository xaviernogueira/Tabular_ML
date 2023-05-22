import warnings
import logging
import numpy as np
from tabular_ml.base import (
    OptunaRangeDict,
)
from typing import (
    Dict,
    Literal,
    Optional,
)

# TODO: add more metric directions
MetricDirections: Dict[str, Literal['minimize', 'maximize']] = {
    'mean_absolute_error': 'minimize',
    'log_loss': 'minimize',
    'r2_score': 'maximize',
}


def get_metric_direction(
    metric_function: callable,
) -> Literal['minimize', 'maximize']:
    """Returns metric function direction"""
    # return value based on metric_function
    if metric_function.__name__ not in MetricDirections.keys():
        logging.warn(
            f'Optimal direction cannot be inferred from metric_function: '
            f'{metric_function.__name__}. Please set param:direction to '
            f'maximize or minimize. Default is minimize!',
        )
        return 'minimize'
    return MetricDirections[metric_function.__name__]


def get_adjusted_score(
    test_losses: np.ndarray,
    metric_function: callable,
) -> float:
    """Gets standard deviation adjusted score.

    NOTE: Behavior depends on eval function directionality.
    """
    # get loss mean/std
    mean = np.mean(np.array(test_losses))
    std = np.std(np.array(test_losses))
    direction = get_metric_direction(metric_function)

    if direction == 'minimize':
        return mean + std
    else:
        return mean - std


def get_ensemble_prediction(
    model_predictions_dict: Dict[str, np.ndarray],
) -> float:
    """Combines numpy array predictions"""
    return np.mean(
        list(model_predictions_dict.values()),
        axis=0,
    )


def get_optuna_ranges(
    default_optuna_ranges: OptunaRangeDict,
    custom_optuna_ranges: Optional[OptunaRangeDict] = None,
) -> OptunaRangeDict:
    """Get the optuna ranges for the model (and/or updates them).

    NOTE: This should be called at the top of the objective function.

    Arguments:
        default_optuna_ranges: The default optuna ranges for the model.
        custom_optuna_ranges: The custom optuna ranges for the model.
    Return:
        The optuna ranges for the model optimization.
    """

    # return base ranges if no custom ranges
    if not custom_optuna_ranges:
        return default_optuna_ranges

    # update the base ranges with the custom ranges if desired
    new_optuna_ranges = default_optuna_ranges.copy()
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
