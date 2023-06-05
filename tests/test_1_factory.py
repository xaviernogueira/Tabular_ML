"""Tests the factory implementation."""
import typing
import tabular_ml


def test_general_factory() -> None:
    all_models_dict = tabular_ml.ModelFactory.get_all_models()
    assert isinstance(all_models_dict, dict)
    keys = list(all_models_dict.keys())
    for key in keys:
        assert key in typing.get_args(tabular_ml.base.ModelTypes)


def test_regression_models() -> None:
    """Tests that regression models are correctly loaded."""

    regression_models = tabular_ml.ModelFactory.get_regression_models()
    assert isinstance(regression_models, list)
    assert len(regression_models) > 0
    assert isinstance(regression_models[0], str)
    first_model_obj = tabular_ml.ModelFactory.get_regression_model(
        regression_models[0],
    )
    assert issubclass(first_model_obj, tabular_ml.base.MLModel)


def test_classification_models() -> None:
    """Tests that classification models are correctly loaded."""

    classification_models = tabular_ml.ModelFactory.get_classification_models()
    assert isinstance(classification_models, list)
    assert len(classification_models) > 0
    assert isinstance(classification_models[0], str)
    first_model_obj = tabular_ml.ModelFactory.get_classification_model(
        classification_models[0],
    )
    assert issubclass(first_model_obj, tabular_ml.base.MLModel)


def test_factory_methods() -> None:
    """Test unregister_model and register_model methods."""

    # get a copy of a model class
    model_name = tabular_ml.ModelFactory.get_regression_models()[0]
    model = tabular_ml.ModelFactory.get_regression_model(model_name)

    # unregister the model
    tabular_ml.ModelFactory.unregister_model(model_name)
    assert model_name not in tabular_ml.ModelFactory.get_regression_models()

    # re-register the model
    tabular_ml.ModelFactory.register_model(model)
    assert model_name in tabular_ml.ModelFactory.get_regression_models()
