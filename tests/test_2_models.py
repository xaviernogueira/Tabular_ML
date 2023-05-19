"""Test using the MLModel implementation on sklearn toy data.
"""
import typing
import pytest
import numpy as np
import pandas as pd
import sklearn
import tabular_ml
from pathlib import Path

print(tabular_ml.ModelFactory.get_all_models())

# TODO: update to sklearn toy datasets later


def model_test(
    model: tabular_ml.base.MLModel,
    train: pd.DataFrame,
    test: pd.DataFrame,
    pred_col: str,
) -> None:
    # check that we can train a model instance
    expected_return_type1 = typing.get_type_hints(
        model.train_model,
    )['return']
    trained_model = model.train_model(
        x_train=train.drop(columns=[pred_col]),
        y_train=train[pred_col],
        model_params={},
    )
    assert isinstance(trained_model, expected_return_type1)

    # check the type hint
    expected_return_type2 = typing.get_type_hints(
        model.make_predictions,
    )['return']
    assert expected_return_type2 == np.ndarray

    # check that we can make predictions
    preds = model.make_predictions(
        trained_model,
        x_test=test.drop(columns=[pred_col]),
    )
    assert isinstance(preds, expected_return_type2)


def test_regression_models() -> None:
    """Tests all regression models."""
    regression_models = tabular_ml.ModelFactory.get_regression_models()

    # get test data
    if Path.cwd().name == 'tests':
        test_data_path = Path.cwd() / 'us_counties_metrics.parquet'
    else:
        test_data_path = Path.cwd() / 'tests/us_counties_metrics.parquet'
    data = pd.read_parquet(
        test_data_path,
    ).fillna(0).iloc[:100]

    train, test = sklearn.model_selection.train_test_split(data)
    pred_col = 'gdp_growth_rate%_2001to2019'

    # test each regression model implementation
    for model_name in regression_models:
        model = tabular_ml.ModelFactory.get_regression_model(model_name)

        model_test(
            model,
            train,
            test,
            pred_col,
        )


def test_classification_models() -> None:
    classification_models = tabular_ml.ModelFactory.get_classification_models()

    # get test data
    if Path.cwd().name == 'tests':
        test_data_path = Path.cwd() / 'us_counties_metrics.parquet'
    else:
        test_data_path = Path.cwd() / 'tests/us_counties_metrics.parquet'
    data = pd.read_parquet(
        test_data_path,
    ).fillna(0).iloc[:100]

    # make a categorical column
    data['pred_col'] = np.round(data.gini_index.values, 0).astype(int)

    train, test = sklearn.model_selection.train_test_split(data)
    pred_col = 'pred_col'

    # test each regression model implementation
    for model_name in classification_models:
        model = tabular_ml.ModelFactory.get_classification_model(model_name)

        model_test(
            model,
            train,
            test,
            pred_col,
        )
