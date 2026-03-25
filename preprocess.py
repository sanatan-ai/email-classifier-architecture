import pandas as pd
import numpy as np
import re
import string
from Config import Config
from deep_translator import GoogleTranslator
from langdetect import detect
import random

seed = 0
random.seed(seed)
np.random.seed(seed)

# Noise words to remove from text (Activity 4)
NOISE_WORDS = [
    'thank you for your email',
    'thank you for contacting us',
    'dear customer',
    'kind regards',
    'best regards',
    'yours sincerely',
    'please find below',
    'hope you are well',
    'i hope this email finds you well',
    'do not hesitate to contact us',
    'feel free to contact us',
    'we apologize for the inconvenience',
]


# =============================================================================
# Activity 1: Data Selection — Load data and drop irrelevant columns
# =============================================================================

def get_input_data() -> pd.DataFrame:
    """
    Loads the dataset specified in Config.DATA_PATH.
    To switch datasets, update Config.DATA_PATH only — no code changes needed.
    """
    df = pd.read_csv(Config.DATA_PATH, encoding='utf-8-sig')

    # Drop unnamed/empty columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Select only the columns we need
    required_cols = [
        'Ticket id',
        Config.TICKET_SUMMARY,
        Config.INTERACTION_CONTENT,
        'Type 1',
        'Type 2',
        'Type 3',
        'Type 4'
    ]
    df = df[required_cols]

    # Rename Type columns to aliases defined in Config
    df = df.rename(columns={
        'Type 1': Config.GROUPED,
        'Type 2': 'y2',
        'Type 3': 'y3',
        'Type 4': 'y4'
    })

    # Drop rows where y2 is missing (primary label must exist)
    df = df.dropna(subset=['y2'])

    # Fill missing y3 and y4 with 'Others'
    df['y3'] = df['y3'].fillna('Others')
    df['y4'] = df['y4'].fillna('Others')

    print(f"[Data Selection] Loaded {len(df)} records from {Config.DATA_PATH}")
    return df


# =============================================================================
# Activity 2: Data Grouping — Group interactions by Ticket ID
# =============================================================================

def de_duplication(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups conversation interactions by Ticket ID.
    All interaction content belonging to the same ticket is
    concatenated into a single record to avoid data leakage.
    """
    # Aggregate interaction content by ticket
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].astype(str)
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].astype(str)

    # Group by Ticket id — concatenate text, keep first label values
    agg_dict = {
        Config.INTERACTION_CONTENT: ' '.join,
        Config.TICKET_SUMMARY: 'first',
        Config.GROUPED: 'first',
        'y2': 'first',
        'y3': 'first',
        'y4': 'first'
    }

    df = df.groupby('Ticket id', as_index=False).agg(agg_dict)

    print(f"[De-duplication] After grouping by ticket: {len(df)} unique tickets")
    return df


# =============================================================================
# Activity 3: Multi-language Handling — Translate all text to English
# =============================================================================

def detect_language(text: str) -> str:
    """
    Detects the language of a given text string.
    """
    try:
        return detect(text)
    except Exception:
        return 'en'


def translate_to_en(texts: list) -> list:
    """
    Translates a list of texts to English.
    Skips translation for texts already in English.
    """
    translated = []
    for text in texts:
        try:
            text = str(text)
            lang = detect_language(text)
            if lang != 'en':
                result = GoogleTranslator(source='auto', target='en').translate(text)
                translated.append(result if result else text)
            else:
                translated.append(text)
        except Exception:
            translated.append(text)  # fallback: keep original

    print(f"[Translation] Translated {len(translated)} records to English")
    return translated


# =============================================================================
# Activity 4: Noise Removal — Clean text data
# =============================================================================

def noise_remover(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes noise from the Interaction Content and Ticket Summary columns.
    Noise includes:
    - Common email phrases (thank you, dear customer, etc.)
    - Email addresses and URLs
    - Special characters and extra whitespace
    - Punctuation
    """

    def clean_text(text: str) -> str:
        text = str(text).lower()

        # Remove noise phrases
        for phrase in NOISE_WORDS:
            text = text.replace(phrase.lower(), ' ')

        # Remove email addresses
        text = re.sub(r'\S+@\S+', ' ', text)

        # Remove URLs
        text = re.sub(r'http\S+|www\S+', ' ', text)

        # Remove phone-like patterns (e.g. *****(PHONE))
        text = re.sub(r'\*+\([A-Z]+\)', ' ', text)

        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].apply(clean_text)
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].apply(clean_text)

    print(f"[Noise Removal] Cleaned text in {len(df)} records")
    return df


# =============================================================================
# Activity 5: Multi-level Data Handling — Combine Summary + Content
# =============================================================================

def combine_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines Ticket Summary and Interaction Content into a single
    text column for classification. This handles multi-level text
    data by merging both fields into one representation.
    """
    df[Config.INTERACTION_CONTENT] = (
            df[Config.TICKET_SUMMARY].astype(str)
            + ' '
            + df[Config.INTERACTION_CONTENT].astype(str)
    )
    print(f"[Multi-level] Combined Ticket Summary + Interaction Content")
    return df
#Methods related to data loading and all pre-processing steps will go here