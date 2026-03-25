import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import random

seed = 0
random.seed(seed)
np.random.seed(seed)


class SVM(BaseModel):
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray) -> None:
        """
        Initialises the SVM classifier.
        """
        super(SVM, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.mdl = SVC(
            kernel='linear',          # Linear kernel works well for text classification
            C=1.0,                    # Regularisation parameter
            class_weight='balanced',  # Handle class imbalance
            random_state=seed,
            probability=True          # Enable probability estimates
        )
        self.predictions = None
        self.data_transform()

    def train(self, data) -> None:
        """
        Trains the SVM model on training data.
        """
        self.mdl = self.mdl.fit(data.X_train, data.y_train)
        print(f"[SVM] Training complete.")

    def predict(self, X_test: np.ndarray) -> None:
        """
        Generates predictions on test data.
        """
        self.predictions = self.mdl.predict(X_test)
        print(f"[SVM] Predictions generated.")

    def print_results(self, data) -> None:
        """
        Prints the classification report comparing predictions against true test labels.
        """
        print(classification_report(data.y_test, self.predictions))

    def data_transform(self) -> None:
        """
        SVM requires no additional data transformation beyond what is handled in data_model.py.
        """
        ...