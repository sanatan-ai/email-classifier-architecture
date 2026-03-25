import numpy as np
import pandas as pd
from Config import Config
from model.randomforest import RandomForest
from model.svm import SVM
from model.naive_bayes import NaiveBayes
from model.logistic_regression import LogisticRegression


# =============================================================================
# Activity 9 & 10: Model Selection, Training and Testing
# =============================================================================

# Registry of all available models
# To add a new model: import it above and add an entry here
MODEL_REGISTRY = {
    'random_forest':        RandomForest,
    'svm':                  SVM,
    'naive_bayes':          NaiveBayes,
    'logistic_regression':  LogisticRegression,
}


def get_model(model_name: str, embeddings: np.ndarray, y: np.ndarray):
    """
    Factory function — returns an instance of the requested model.
    All models are accessed through the same BaseModel interface (Feature 3).
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"[Modelling] Unknown model '{model_name}'. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )

    model_class = MODEL_REGISTRY[model_name]
    print(f"[Modelling] Selected model: {model_name}")
    return model_class(
        model_name=model_name,
        embeddings=embeddings,
        y=y
    )


def model_predict(data, df: pd.DataFrame, name: str):
    """
    Full pipeline for a single model:
    1. Selects the model by name
    2. Trains on training data
    3. Predicts on test data
    4. Prints evaluation results

    Interfaces only with BaseModel methods — hides all
    model-level differences from this controller (Feature 3).
    """
    # Step 1: Select model (Activity 9)
    model = get_model(name, data.get_embeddings(), data.get_y())

    # Step 2: Train (Activity 10)
    print(f"[Modelling] Training {name}...")
    model.train(data)

    # Step 3: Predict (Activity 10)
    print(f"[Modelling] Running predictions for {name}...")
    model.predict(data.get_X_test())

    # Step 4: Evaluate
    model_evaluate(model, data)


def model_evaluate(model, data):
    """
    Prints the classification results using the model's print_results() method — consistent across all models (Feature 3).
    """
    print(f"\n[Evaluation] Results for: {model.model_name}")
    print("-" * 50)
    model.print_results(data)


def run_all_models(data, df: pd.DataFrame):
    """
    Convenience function — runs the full pipeline for ALL models in the registry sequentially.
    """
    print("\n" + "=" * 50)
    print("Running all models")
    print("=" * 50)

    for model_name in MODEL_REGISTRY:
        print(f"\n>>> Model: {model_name.upper()}")
        model_predict(data, df, model_name)