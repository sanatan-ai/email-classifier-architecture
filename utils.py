import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from Config import Config
import random
import os

seed = 0
random.seed(seed)
np.random.seed(seed)


# =============================================================================
# Label Encoding Utilities
# =============================================================================

def encode_labels(df: pd.DataFrame, col: str) -> tuple:
    """
    Encodes string labels in a column into numeric values.
    Used before passing labels into ML models.
    """
    le = LabelEncoder()
    encoded = le.fit_transform(df[col].astype(str))
    print(f"[Utils] Encoded column '{col}' → classes: {list(le.classes_)}")
    return encoded, le


def decode_labels(encoded: np.ndarray, le: LabelEncoder) -> np.ndarray:
    """
    Decodes numeric labels back to original string values.
    """
    return le.inverse_transform(encoded)


# =============================================================================
# Data Validation Utilities
# =============================================================================

def check_min_class_instances(df: pd.DataFrame, col: str, min_count: int = 3) -> pd.DataFrame:
    """
    Removes rows belonging to classes that have fewer than
    min_count instances in the given column.
    Used to avoid training on classes with insufficient data.
    """
    class_counts = df[col].value_counts()
    valid_classes = class_counts[class_counts >= min_count].index
    removed = class_counts[class_counts < min_count]

    if not removed.empty:
        print(f"[Utils] Removing classes with fewer than {min_count} instances from '{col}':")
        for cls, count in removed.items():
            print(f"        - '{cls}': {count} instance(s)")

    filtered = df[df[col].isin(valid_classes)].reset_index(drop=True)
    print(f"[Utils] Records after class filtering: {len(filtered)}")
    return filtered


def validate_dataframe(df: pd.DataFrame, required_cols: list) -> bool:
    """
    Validates that a DataFrame contains all required columns.
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"[Utils] Missing required columns: {missing}")
    print(f"[Utils] DataFrame validation passed. Shape: {df.shape}")
    return True


# =============================================================================
# Output Utilities
# =============================================================================

def ensure_dir(path: str) -> None:
    """
    Creates a directory if it does not already exist.
    """
    os.makedirs(path, exist_ok=True)


def print_class_distribution(df: pd.DataFrame, col: str) -> None:
    """
    Prints the class distribution for a given column.
    Useful for inspecting label balance before modelling.
    """
    print(f"\n[Utils] Class distribution for '{col}':")
    counts = df[col].value_counts()
    for cls, count in counts.items():
        pct = (count / len(df)) * 100
        print(f"        {cls:<40} {count:>4} ({pct:.1f}%)")


def string2any(value: str):
    """
    Attempts to convert a string to its most appropriate Python type.
    Used by BaseModel.build() to parse config values.
    """
    if value.lower() == 'true':
        return True
    if value.lower() == 'false':
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
