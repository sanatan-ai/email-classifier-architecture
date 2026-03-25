import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from modelling.data_model import Data
from modelling.modelling import get_model
from utils import print_class_distribution, check_min_class_instances
from Config import Config

seed = 0


# =============================================================================
# Design Decision 2: Hierarchical Modelling
# =============================================================================
# Model instances are chained where the OUTPUT of the previous instance
# FILTERS the input data for the next instance:
#
# Level 1: One model instance classifies y2
#          - produces N classes (e.g. Others, Problem/Fault, Suggestion)
#
# Level 2: For each class in y2, filter data and train a new model on y3
#          - produces N model instances (one per y2 class)
#          - each instance is called a Filter Set A
#
# Level 3: For each Filter Set A, filter data by y3 class and train on y4
#          - produces N x M model instances (one per y2 x y3 combination)
#
# Total model instances = 1 + |y2_classes| + |y2_classes| x |y3_classes|
# =============================================================================


def _make_data_container(X: np.ndarray,
                         y: np.ndarray,
                         test_size: float = 0.2):
    """
    Builds a lightweight train/test container from filtered X and y.
    Used at each hierarchical level to prepare data for a model instance.
    """
    if len(np.unique(y)) < 2 or len(y) < 6:
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed
    )

    class HierarchicalData:
        pass

    container = HierarchicalData()
    container.X_train = X_train
    container.X_test  = X_test
    container.y_train = y_train
    container.y_test  = y_test
    container.embeddings = X
    container.y = y

    return container


def _filter_data(data: Data,
                 df: pd.DataFrame,
                 filter_col: str,
                 filter_val: str):
    """
    Filters the dataset to rows where filter_col == filter_val.
    Returns filtered X and df for the next hierarchical level.
    """
    mask = df[filter_col] == filter_val
    filtered_df = df[mask].reset_index(drop=True)
    filtered_X  = data.embeddings[mask.values]
    return filtered_X, filtered_df


def run_hierarchical_model(data: Data,
                           df: pd.DataFrame,
                           model_name: str) -> None:
    """
    Runs the full Hierarchical Modelling pipeline for a given model.

    Level 1: Train one model instance on y2 using full dataset
    Level 2: For each y2 class → filter data → train new instance on y3
    Level 3: For each (y2 class, y3 class) → filter data → train new instance on y4
    """
    print("\n" + "=" * 60)
    print(f"DESIGN DECISION 2: Hierarchical Modelling — Model: {model_name.upper()}")
    print("=" * 60)

    results = {}  # Store results for summary

    # -------------------------------------------------------------------------
    # LEVEL 1 — Classify y2 using the full dataset
    # -------------------------------------------------------------------------
    print("\n>>> LEVEL 1: Classifying y2 (full dataset)")
    print_class_distribution(df, 'y2')

    y2_labels = df['y2'].values
    level1_data = _make_data_container(data.embeddings, y2_labels)

    if level1_data is None:
        print("[Hierarchical] Insufficient data for Level 1. Aborting.")
        return

    model_l1 = get_model(model_name, level1_data.embeddings, level1_data.y)
    model_l1.train(level1_data)
    model_l1.predict(level1_data.X_test)

    print(f"\n[Level 1] Results — y2 classification:")
    print("-" * 50)
    print(classification_report(
        level1_data.y_test,
        model_l1.predictions,
        zero_division=0
    ))

    results['level_1'] = {
        'target': 'y2',
        'model': model_l1
    }

    # Get unique y2 classes from the full dataset
    y2_classes = df['y2'].dropna().unique()
    print(f"\n[Hierarchical] y2 classes found: {list(y2_classes)}")

    # -------------------------------------------------------------------------
    # LEVEL 2 — For each y2 class, filter data and classify y3
    # -------------------------------------------------------------------------
    print("\n>>> LEVEL 2: Classifying y3 (filtered per y2 class)")

    results['level_2'] = {}

    for y2_class in y2_classes:
        print(f"\n  [Filter Set A] y2 = '{y2_class}'")

        # Filter data for this y2 class
        filtered_X, filtered_df = _filter_data(data, df, 'y2', y2_class)

        if len(filtered_df) < 6:
            print(f"  [Hierarchical] Skipping y2='{y2_class}' — only {len(filtered_df)} records.")
            continue

        # Remove y3 classes with too few instances in this filtered subset
        filtered_df = check_min_class_instances(filtered_df, 'y3', min_count=3)

        if len(filtered_df) < 6:
            print(f"  [Hierarchical] Skipping y2='{y2_class}' after class filtering.")
            continue

        print_class_distribution(filtered_df, 'y3')

        y3_labels = filtered_df['y3'].values
        # Re-align X to filtered_df indices
        mask_l2 = df['y2'] == y2_class
        X_l2 = data.embeddings[mask_l2.values]

        # Further align X to post-class-filter indices
        X_l2 = X_l2[:len(filtered_df)]

        level2_data = _make_data_container(X_l2, y3_labels)

        if level2_data is None:
            print(f"  [Hierarchical] Skipping — insufficient unique classes in y3.")
            continue

        model_l2 = get_model(model_name, level2_data.embeddings, level2_data.y)
        model_l2.train(level2_data)
        model_l2.predict(level2_data.X_test)

        print(f"\n  [Level 2] Results — y3 classification (y2='{y2_class}'):")
        print("  " + "-" * 48)
        print(classification_report(
            level2_data.y_test,
            model_l2.predictions,
            zero_division=0
        ))

        results['level_2'][y2_class] = {
            'target': 'y3',
            'filter': f"y2={y2_class}",
            'model': model_l2
        }

        # ---------------------------------------------------------------------
        # LEVEL 3 — For each y3 class within this y2 class, classify y4
        # ---------------------------------------------------------------------
        y3_classes = filtered_df['y3'].dropna().unique()
        print(f"\n  >>> LEVEL 3: Classifying y4 under y2='{y2_class}'")
        print(f"      y3 classes found: {list(y3_classes)}")

        results['level_2'][y2_class]['level_3'] = {}

        for y3_class in y3_classes:
            print(f"\n    [Filter Set B] y2='{y2_class}' + y3='{y3_class}'")

            # Filter within the already-filtered Level 2 subset
            mask_l3 = (df['y2'] == y2_class) & (df['y3'] == y3_class)
            X_l3 = data.embeddings[mask_l3.values]
            df_l3 = df[mask_l3].reset_index(drop=True)

            if len(df_l3) < 6:
                print(f"    [Hierarchical] Skipping — only {len(df_l3)} records.")
                continue

            df_l3 = check_min_class_instances(df_l3, 'y4', min_count=3)

            if len(df_l3) < 6:
                print(f"    [Hierarchical] Skipping after y4 class filtering.")
                continue

            X_l3 = X_l3[:len(df_l3)]
            print_class_distribution(df_l3, 'y4')

            y4_labels = df_l3['y4'].values
            level3_data = _make_data_container(X_l3, y4_labels)

            if level3_data is None:
                print(f"    [Hierarchical] Skipping — insufficient unique classes in y4.")
                continue

            model_l3 = get_model(model_name, level3_data.embeddings, level3_data.y)
            model_l3.train(level3_data)
            model_l3.predict(level3_data.X_test)

            print(f"\n    [Level 3] Results — y4 (y2='{y2_class}', y3='{y3_class}'):")
            print("    " + "-" * 46)
            print(classification_report(
                level3_data.y_test,
                model_l3.predictions,
                zero_division=0
            ))

            results['level_2'][y2_class]['level_3'][y3_class] = {
                'target': 'y4',
                'filter': f"y2={y2_class}, y3={y3_class}",
                'model': model_l3
            }

    _print_hierarchy_summary(results, model_name)


def _print_hierarchy_summary(results: dict, model_name: str) -> None:
    """
    Prints a summary of all model instances created during
    hierarchical modelling — showing the tree structure.
    """
    print("\n" + "=" * 60)
    print(f"HIERARCHY SUMMARY — {model_name.upper()}")
    print("=" * 60)
    print(f"  Level 1 → 1 model instance (y2 classification)")

    total_l2 = len(results.get('level_2', {}))
    print(f"  Level 2 → {total_l2} model instance(s) (y3 per y2 class)")

    total_l3 = sum(
        len(v.get('level_3', {}))
        for v in results.get('level_2', {}).values()
    )
    print(f"  Level 3 → {total_l3} model instance(s) (y4 per y2+y3 combination)")
    print(f"  {'─' * 40}")
    print(f"  Total model instances: {1 + total_l2 + total_l3}")
    print("=" * 60)


def run_all_hierarchical_models(data: Data, df: pd.DataFrame) -> None:
    """
    Runs Hierarchical Modelling pipeline for ALL models in the registry.
    """
    from modelling.modelling import MODEL_REGISTRY

    for model_name in MODEL_REGISTRY:
        run_hierarchical_model(data, df, model_name)