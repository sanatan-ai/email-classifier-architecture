from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class BaseModel(ABC):
    def __init__(self) -> None:
        ...

    @abstractmethod
    def train(self, data) -> None:
        """
        Train the model using ML Models for Multi-class and multi-label classification.
        """
        ...

    @abstractmethod
    def predict(self, X_test) -> None:
        """
        Run predictions on the test set.
        """
        ...

    @abstractmethod
    def print_results(self, data) -> None:
        """
        Print evaluation results (e.g., classification report).
        """
        ...

    @abstractmethod
    def data_transform(self) -> None:
        """
        Perform any model-specific data transformation before training.
        """
        ...

    def build(self, values={}):
        self.__dict__.update(values)
        return self