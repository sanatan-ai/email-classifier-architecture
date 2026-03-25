import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from Config import Config
import random

seed = 0
random.seed(seed)
np.random.seed(seed)

# =============================================================================
# Activity 6: Textual Data Representation — Convert text to numeric (TF-IDF)
# =============================================================================

# Global vectorizer instance — fit once, reuse for consistency
tfidf_vectorizer = TfidfVectorizer(
    max_features=1000,  # Keep top 1000 features
    stop_words='english',  # Remove common English stopwords
    ngram_range=(1, 2),  # Use unigrams and bigrams
    min_df=2,  # Ignore terms appearing in fewer than 2 documents
    sublinear_tf=True  # Apply log normalization to term frequency
)


def get_tfidf_embd(df: pd.DataFrame) -> np.ndarray:
    """
    Converts the cleaned Interaction Content text into TF-IDF numeric embeddings.
    Combines Ticket Summary and Interaction Content as the input text.

    The vectorizer is fit on the entire dataset here so that the vocabulary
    is consistent across training and testing splits (split happens later
    in data_model.py).

    """
    # Use Interaction Content as the primary text input
    # (already combined with Ticket Summary in preprocess.py - Activity 5)
    corpus = df[Config.INTERACTION_CONTENT].astype(str).tolist()

    # Fit and transform the corpus
    X = tfidf_vectorizer.fit_transform(corpus)

    # Convert sparse matrix to dense numpy array
    X = X.toarray()

    print(f"[Embeddings] TF-IDF matrix shape: {X.shape}")
    print(f"[Embeddings] Vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")

    return X


def get_vectorizer() -> TfidfVectorizer:
    """
    Returns the fitted TF-IDF vectorizer instance.
    Useful if transform-only is needed on new data.
    """
    return tfidf_vectorizer
#Methods related to converting text in into numeric representation and then returning numeric representation may go here