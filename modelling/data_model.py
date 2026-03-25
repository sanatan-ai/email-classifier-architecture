import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import Config
import random

seed = 0
random.seed(seed)
np.random.seed(seed)

# Minimum number of instances required for a class to be kept
MIN_CLASS_INSTANCES = 3


class Data:
    def __init__(self,
                 X: np.ndarray,
                 df: pd.DataFrame) -> None:
        """
        Encapsulates all data required for model training and testing.
        Implements Feature 2: consistent input format across all ML models.
        """

        # --- Activity 7: Remove classes with too few instances ---
        # Work on the primary classification column (y2)
        class_counts = df[Config.CLASS_COL].value_counts()
        valid_classes = class_counts[class_counts >= MIN_CLASS_INSTANCES].index
        mask = df[Config.CLASS_COL].isin(valid_classes)

        # Apply mask to both X and df
        X = X[mask]
        df = df[mask].reset_index(drop=True)

        # --- Store full embeddings and dataframe ---
        self.embeddings = X
        self.df = df

        # --- Build y: use CLASS_COL (y2) as primary target for standard modelling ---
        # TYPE_COLS (y2, y3, y4) are kept in df for strategies (chained/hierarchical)
        self.y = df[Config.CLASS_COL].values  # shape: (n_samples,) — single column

        # --- Activity 8: Train/Test Split (80/20) ---
        (self.X_train,
         self.X_test,
         self.y_train,
         self.y_test,
         self.train_df,
         self.test_df) = train_test_split(
            X,
            self.y,
            df,
            test_size=0.2,
            random_state=seed
        )

    # --- Getters ---
    def get_y(self):
        return self.y

    def get_X_train(self):
        return self.X_train

    def get_X_test(self):
        return self.X_test

    def get_y_train(self):
        return self.y_train

    def get_y_test(self):
        return self.y_test

    def get_train_df(self):
        return self.train_df

    def get_test_df(self):
        return self.test_df

    def get_embeddings(self):
        return self.embeddings

    def get_filtered_data(self, filter_col: str, filter_val: str):
        """
        Returns a new Data-like filtered view for Hierarchical Modelling (Design Decision 2).
        Filters the dataset where filter_col == filter_val.
        """
        mask = self.df[filter_col] == filter_val
        filtered_df = self.df[mask].reset_index(drop=True)
        filtered_X = self.embeddings[mask]

        # Return a new Data object built from the filtered subset
        return Data(filtered_X, filtered_df)