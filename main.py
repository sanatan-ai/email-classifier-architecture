import numpy as np
import pandas as pd
import random

from preprocess import (
    get_input_data,
    de_duplication,
    noise_remover,
    translate_to_en,
    combine_text_fields
)
from embeddings import get_tfidf_embd
from modelling.data_model import Data
from modelling.modelling import model_predict, run_all_models
from strategies.chained import run_chained_model, run_all_chained_models
from strategies.hierarchical import run_hierarchical_model, run_all_hierarchical_models
from utils import print_class_distribution, validate_dataframe
from Config import Config

seed = 0
random.seed(seed)
np.random.seed(seed)


# =============================================================================
# Pipeline Step Functions
# =============================================================================

def load_data() -> pd.DataFrame:
    """
    Load and select data from the configured dataset.
    Dataset path is controlled via Config.DATA_PATH.
    """
    df = get_input_data()
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline.
      - Group interactions by ticket (de-duplication)
      - Translate all text to English
      - Remove noise from text
      - Combine Ticket Summary + Interaction Content
    """
    # Activity 2: Group by ticket ID
    df = de_duplication(df)

    # Activity 3: Translate to English
    df[Config.TICKET_SUMMARY] = translate_to_en(
        df[Config.TICKET_SUMMARY].tolist()
    )
    df[Config.INTERACTION_CONTENT] = translate_to_en(
        df[Config.INTERACTION_CONTENT].tolist()
    )

    # Activity 4: Remove noise
    df = noise_remover(df)

    # Activity 5: Combine text fields for classification
    df = combine_text_fields(df)

    return df


def get_embeddings(df: pd.DataFrame):
    """
    Convert cleaned text to TF-IDF numeric representation.
    """
    X = get_tfidf_embd(df)
    return X, df


def get_data_object(X: np.ndarray, df: pd.DataFrame) -> Data:
    """
    Encapsulate data into a Data object.
      - Remove classes with too few instances
      - Train/test split
    """
    return Data(X, df)


def perform_modelling(data: Data, df: pd.DataFrame, name: str) -> None:
    """
    Train and evaluate a single model by name.
    Model name must be a key in MODEL_REGISTRY in modelling.py.

    Available models:
        'random_forest'
        'svm'
        'naive_bayes'
        'logistic_regression'
    """
    model_predict(data, df, name)


def perform_chained_modelling(data: Data, model_name: str) -> None:
    """
    Design Decision 1: Chained Multi-Output Classification.
    Evaluates one model instance against progressively combined
    target variables: y2 → y2+y3 → y2+y3+y4.
    """
    run_chained_model(data, model_name)


def perform_hierarchical_modelling(data: Data, df: pd.DataFrame,
                                   model_name: str) -> None:
    """
    Design Decision 2: Hierarchical Modelling.
    Chains model instances where each level's output filters
    the input data for the next level.
    """
    run_hierarchical_model(data, df, model_name)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':

    print("\n" + "=" * 60)
    print("EMAIL CLASSIFIER — PIPELINE START")
    print("=" * 60)

    # ------------------------------------------------------------------
    # STAGE 1: Preprocessing (Activities 1–5)
    # ------------------------------------------------------------------
    print("\n[Stage 1] Loading and preprocessing data...")
    df = load_data()
    df = preprocess_data(df)

    # Ensure text columns are clean unicode strings
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')

    # Validate required columns are present
    validate_dataframe(df, Config.TYPE_COLS + [
        Config.TICKET_SUMMARY,
        Config.INTERACTION_CONTENT
    ])

    # Print label distributions before modelling
    for col in Config.TYPE_COLS:
        print_class_distribution(df, col)

    # ------------------------------------------------------------------
    # STAGE 2: Embeddings (Activity 6)
    # ------------------------------------------------------------------
    print("\n[Stage 2] Generating TF-IDF embeddings...")
    X, df = get_embeddings(df)

    # ------------------------------------------------------------------
    # STAGE 3: Data Object (Activities 7–8)
    # ------------------------------------------------------------------
    print("\n[Stage 3] Building data object (class filtering + train/test split)...")
    data = get_data_object(X, df)

    # ------------------------------------------------------------------
    # STAGE 4: Standard Modelling (Activities 9–10)
    # ------------------------------------------------------------------
    print("\n[Stage 4] Standard multi-label classification...")

    # Run all models
    run_all_models(data, df)

    # ------------------------------------------------------------------
    # STAGE 5: Design Decision 1 — Chained Multi-Output
    # ------------------------------------------------------------------
    print("\n[Stage 5] Design Decision 1 — Chained Multi-Output Classification...")

    # Run all models
    run_all_chained_models(data)

    # ------------------------------------------------------------------
    # STAGE 6: Design Decision 2 — Hierarchical Modelling
    # ------------------------------------------------------------------
    print("\n[Stage 6] Design Decision 2 — Hierarchical Modelling...")

    # Run all models
    run_all_hierarchical_models(data, df)

    print("\n" + "=" * 60)
    print("EMAIL CLASSIFIER — PIPELINE COMPLETE")
    print("=" * 60)