import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
import random

seed = 0
random.seed(seed)
np.random.seed(seed)


class NaiveBayes(BaseModel):
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray) -> None:
        """
        Initialises the Naive Bayes classifier.
        Uses MultinomialNB which is well suited for text classification.
        """
        super(NaiveBayes, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.mdl = MultinomialNB(
            alpha=1.0,       # Laplace smoothing to handle zero probabilities
            fit_prior=True   # Learn class prior probabilities from training data
        )
        self.scaler = MinMaxScaler()  # Ensures non-negative input for MultinomialNB
        self.predictions = None
        self.data_transform()

    def train(self, data) -> None:
        """
        Trains the Naive Bayes model on training data.
        Scales X_train to non-negative range before fitting.
        """
        X_train_scaled = self.scaler.fit_transform(data.X_train)
        self.mdl = self.mdl.fit(X_train_scaled, data.y_train)
        print(f"[NaiveBayes] Training complete.")

    def predict(self, X_test: np.ndarray) -> None:
        """
        Generates predictions on test data.
        Applies the same scaling used during training.
        """
        X_test_scaled = self.scaler.transform(X_test)
        self.predictions = self.mdl.predict(X_test_scaled)
        print(f"[NaiveBayes] Predictions generated.")

    def print_results(self, data) -> None:
        """
        Prints the classification report comparing predictions against true test labels.
        """
        print(classification_report(data.y_test, self.predictions))

    def data_transform(self) -> None:
        """
        Naive Bayes requires non-negative feature values.
        Scaling is handled inside train() and predict() using the MinMaxScaler fitted on training data only —
        ensuring no data leakage from test set.
        """
        ...