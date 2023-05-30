import warnings
from tabular_ml.base import (
    MLModel,
    ModelTypes,
)
from typing import (
    List,
    Dict,
    get_args,
)


class ModelFactory:
    __registered_models = {
        'regression': {},
        'classification': {},
    }

    @classmethod
    def implemented_model(
        cls,
        args,
    ) -> None:
        """Decorator to register a model implementation."""
        cls.register_model(args)

    @classmethod
    def register_model(
        cls,
        model: MLModel,
    ) -> None:
        """Registers a model."""
        if not issubclass(model, MLModel):
            raise TypeError(
                f'Please provide a valid model implementation! '
                f'Expected a subclass of MLModel, got {type(model)}',
            )
        if model.model_type not in get_args(ModelTypes):
            raise TypeError(
                f'Please provide a valid model type! '
                f'Expected one of {ModelTypes}, got {model.model_type}',
            )
        for func in MLModel.__abstractmethods__:
            assert hasattr(model, func)

        cls.__registered_models[model.model_type][model.__name__] = model

    @classmethod
    def get_all_models(cls) -> Dict[str, List[str]]:
        """Returns all registered models sorted by types."""
        return {
            'regression': list(cls.__registered_models['regression'].keys()),
            'classification': list(cls.__registered_models['classification'].keys()),
        }

    @classmethod
    def get_regression_models(cls) -> List[str]:
        """Returns all registered regression models."""
        return list(cls.__registered_models['regression'].keys())

    @classmethod
    def get_classification_models(cls) -> Dict[str, MLModel]:
        """Returns all registered classification models."""
        return list(cls.__registered_models['classification'].keys())

    @classmethod
    def get_regression_model(cls, model_name: str) -> MLModel:
        """Returns a registered regression model."""
        try:
            return cls.__registered_models['regression'][model_name]
        except KeyError:
            raise KeyError(
                f'Please provide a valid regression model name! '
                f'Expected one of {cls.get_regression_models()}, got {model_name}',
            )

    @classmethod
    def get_classification_model(cls, model_name: str) -> MLModel:
        """Returns a registered classification model."""
        try:
            return cls.__registered_models['classification'][model_name]
        except KeyError:
            raise KeyError(
                f'Please provide a valid classification model name! '
                f'Expected one of {cls.get_classification_models()}, got {model_name}',
            )

    @classmethod
    def get_model(cls, model_name: str) -> MLModel:
        """Returns a registered model (type ambiguous)."""
        if (model_name in cls.get_classification_models()) & \
                (model_name in cls.get_regression_models()):
            warnings.warn(
                f'{model_name} is a registered model for both regression '
                'and classification! This function will return the regression model.',
            )
        try:
            return cls.__registered_models['regression'][model_name]
        except KeyError:
            try:
                return cls.__registered_models['classification'][model_name]
            except KeyError:
                raise KeyError(
                    f'{model_name} is an invalid model name! '
                    f'Use ModelFactory.get_all_models() to see registered models.',
                )
