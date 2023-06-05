"""Test the following functions on sklearn toy data:
    * k_fold_cv()
    * find_optimal_parameters()
"""
import typing
import pytest
import sklearn
import sklearn.datasets
import sklearn.metrics
import pandas as pd
import tabular_ml


def k_fold_test(
    model_names: typing.List[str],
    x_data: pd.DataFrame,
    y_data: pd.Series,
    metric_function: callable,
    metric_function_kwargs: typing.Optional[dict] = None,
) -> None:
    # make model params dict (have them empty)
    model_params = {}
    for model_name in model_names:
        model_params[model_name] = {}

    # run k-fold
    full_results = tabular_ml.k_fold_cv(
        x_data,
        y_data,
        model_names=model_names,
        model_params=model_params,
        n_splits=3,
        random_state=1,
        metric_function=metric_function,
        metric_function_kwargs=metric_function_kwargs,
    )
    assert isinstance(full_results, tabular_ml.KFoldOutput)
    for key, field in tabular_ml.KFoldOutput.__dataclass_fields__.items():
        assert field.name in dir(full_results)

        test_type = typing.get_origin(field.type)
        if not test_type:
            test_type = field.type
        if test_type.__name__ != 'UnionType':
            assert isinstance(getattr(full_results, key), test_type)


def optuna_test(
    model_name: str,
    x_data: pd.DataFrame,
    y_data: pd.Series,
    metric_function: callable,
    custom_optuna_ranges: typing.Optional[dict] = None,
) -> None:

    try:
        out_dict = tabular_ml.find_optimal_parameters(
            model_name,
            x_data,
            y_data,
            metric_function,
            n_trials=2,
            kfolds=2,
            custom_optuna_ranges=custom_optuna_ranges,
        )
        assert isinstance(out_dict, dict)
    except NotImplementedError:
        pass


def test_regression_models() -> None:
    """Tests K-Fold CV and optuna optimization for regression models."""

    # get regression models
    regression_models = tabular_ml.ModelFactory.get_regression_models()

    # get test regression data
    data = sklearn.datasets.load_diabetes(
        as_frame=True,
    ).frame.fillna(0).iloc[:300]

    pred_col = 'target'

    k_fold_test(
        regression_models,
        data.drop(columns=[pred_col]),
        data[pred_col],
        metric_function=sklearn.metrics.r2_score,
    )

    for model_name in regression_models:

        # allow test coverage for custom optuna ranges
        custom_optuna_ranges = None
        if model_name == 'XGBoostRegressionModel':
            custom_optuna_ranges = {
                'early_stopping_rounds': (20, 90),
            }

        optuna_test(
            model_name,
            data.drop(columns=[pred_col]),
            data[pred_col],
            metric_function=sklearn.metrics.r2_score,
            custom_optuna_ranges=custom_optuna_ranges,
        )


def test_classification_models() -> None:
    """Tests K-Fold CV and optuna optimization for classification models."""

    # get classification models
    classification_models = tabular_ml.ModelFactory.get_classification_models()

    # get test classification data
    data = sklearn.datasets.load_iris(
        as_frame=True,
    ).frame.fillna(0).iloc[:300]

    pred_col = 'target'

    # test each classification model implementation
    k_fold_test(
        classification_models,
        data.drop(columns=[pred_col]),
        data[pred_col],
        metric_function=sklearn.metrics.log_loss,
        metric_function_kwargs={'labels': [0, 1, 2]},
    )

    for model_name in classification_models:
        optuna_test(
            model_name,
            data.drop(columns=[pred_col]),
            data[pred_col],
            metric_function=sklearn.metrics.log_loss,
        )


test_regression_models()
