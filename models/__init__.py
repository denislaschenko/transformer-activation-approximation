from config import Model, Approximation
from . import distilbert


def run_experiment(model: Model, approximation: Approximation):
    match model:
        case Model.DISTILBERT:
            return distilbert.run_experiment(approximation)

        case _:
            raise ValueError(f"Unsupported model: {model}")