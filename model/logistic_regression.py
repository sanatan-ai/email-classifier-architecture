import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.metrics import classification_report
import random

seed = 0
random.seed(seed)
np.random.seed(seed)


class LogisticRegression(BaseModel):
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray) -> None:
        """
        Initialises the Logistic Regression classifier.
        """
        super(LogisticRegression, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.mdl = SklearnLR(
            max_iter=1000,            # Sufficient iterations for convergence on text data
            class_weight='balanced',  # Handle class imbalance
            solver='lbfgs',           # Efficient solver for multiclass problems
            multi_class='auto',       # Automatically selects best multiclass strategy
            random_state=seed,
            C=1.0                     # Regularisation — inverse of strength
        )
        self.predictions = None
        self.data_transform()

    def train(self, data) -> None:
        """
        Trains the Logistic Regression model on training data.
        """
        self.mdl = self.mdl.fit(data.X_train, data.y_train)
        print(f"[LogisticRegression] Training complete.")

    def predict(self, X_test: np.ndarray) -> None:
        """
        Generates predictions on test data.
        """
        self.predictions = self.mdl.predict(X_test)
        print(f"[LogisticRegression] Predictions generated.")

    def print_results(self, data) -> None:
        """
        Prints the classification report comparing predictions against true test labels.
        """
        print(classification_report(data.y_test, self.predictions))

    def data_transform(self) -> None:
        """
        Logistic Regression requires no additional data transformation beyond what is handled in data_model.py.
        """
        ...