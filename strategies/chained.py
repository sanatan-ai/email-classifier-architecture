import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from modelling.data_model import Data
from modelling.modelling import get_model
from utils import print_class_distribution
from Config import Config


# =============================================================================
# Design Decision 1: Chained Multi-Output Classification
# =============================================================================
# One model instance is evaluated against progressively combined target
# variables:
#   Chain 1: y2
#   Chain 2: y2 + y3  (concatenated as a single combined label)
#   Chain 3: y2 + y3 + y4
#
# The same model is assessed on its ability to classify each chain level.
# Input X stays the same — only the target Y changes at each chain level.
# =============================================================================


def build_chained_labels(df: pd.DataFrame) -> dict:
    """
    Builds three versions of the target label by progressively
    combining Type columns into a single string label.

    Example:
        Chain 1: 'Problem/Fault'
        Chain 2: 'Problem/Fault_AppGallery-Install/Upgrade'
        Chain 3: 'Problem/Fault_AppGallery-Install/Upgrade_Can't update Apps'
    """
    chains = {
        'chain_1': df['y2'].astype(str),
        'chain_2': df['y2'].astype(str) + '_' + df['y3'].astype(str),
        'chain_3': df['y2'].astype(str) + '_' + df['y3'].astype(str) + '_' + df['y4'].astype(str)
    }

    print(f"\n[Chained] Chain label counts:")
    for chain_name, labels in chains.items():
        print(f"  {chain_name}: {labels.nunique()} unique combined labels")

    return chains


def run_chained_model(data: Data, model_name: str) -> None:
    """
    Runs the full Chained Multi-Output pipeline for a given model.

    For each chain level:
      1. Builds combined target labels
      2. Creates a new Data object with the combined labels as y
      3. Trains the model on X_train with combined y_train
      4. Predicts on X_test
      5. Prints evaluation results for that chain level
    """
    print("\n" + "=" * 60)
    print(f"DESIGN DECISION 1: Chained Multi-Output — Model: {model_name.upper()}")
    print("=" * 60)

    # Build chain labels from full dataframe
    chains = build_chained_labels(data.df)

    for chain_name, combined_labels in chains.items():
        print(f"\n--- {chain_name.upper()} ---")
        print_class_distribution(
            pd.DataFrame({chain_name: combined_labels}),
            chain_name
        )

        # Build a chained Data object with the new combined y
        chained_data = _build_chained_data(data, combined_labels.values)

        # Skip chain if insufficient data after filtering
        if chained_data is None:
            print(f"[Chained] Skipping {chain_name} — insufficient data.")
            continue

        # Instantiate a fresh model for this chain level
        model = get_model(model_name, chained_data.embeddings, chained_data.y)

        # Train
        print(f"[Chained] Training on {chain_name}...")
        model.train(chained_data)

        # Predict
        model.predict(chained_data.X_test)

        # Evaluate
        print(f"\n[Chained] Results for {chain_name} — {model_name.upper()}:")
        print("-" * 50)
        print(classification_report(
            chained_data.y_test,
            model.predictions,
            zero_division=0
        ))


def _build_chained_data(data: Data, combined_y: np.ndarray):
    """
    Builds a lightweight data container for a specific chain level.
    Reuses X splits from the original Data object but replaces y
    with the combined chain labels.
    """
    from sklearn.model_selection import train_test_split
    import random

    seed = 0

    # Filter out classes with fewer than 3 instances in combined_y
    unique, counts = np.unique(combined_y, return_counts=True)
    valid_classes = unique[counts >= 3]

    if len(valid_classes) < 2:
        return None

    mask = np.isin(combined_y, valid_classes)
    X_filtered = data.embeddings[mask]
    y_filtered = combined_y[mask]

    if len(y_filtered) < 10:
        return None

    # Re-split with the combined labels
    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered,
        y_filtered,
        test_size=0.2,
        random_state=seed
    )

    # Return a simple container object
    class ChainedData:
        pass

    chained = ChainedData()
    chained.X_train = X_train
    chained.X_test = X_test
    chained.y_train = y_train
    chained.y_test = y_test
    chained.embeddings = X_filtered
    chained.y = y_filtered

    return chained


def run_all_chained_models(data: Data) -> None:
    """
    Runs Chained Multi-Output pipeline for ALL models in the registry.
    """
    from modelling.modelling import MODEL_REGISTRY

    for model_name in MODEL_REGISTRY:
        run_chained_model(data, model_name)