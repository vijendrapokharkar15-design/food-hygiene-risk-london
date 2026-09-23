"""
Load the enriched data, add engineered features, define the feature list,
and make the fixed train / validation / test split.

Used by notebooks and training scripts:
    from features import load_rated, make_splits, FEATURES
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from enrich_postcodes import PROJECT_ROOT

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TARGET = "fail"
GROUP = "LocalAuthorityName"   # rankings and evaluation happen within each borough
RATINGS = ["0", "1", "2", "3", "4", "5"]

CATEGORICAL = ["BusinessType", "LocalAuthorityName", "area_match_level"]
NUMERIC = [
    "imd_rank", "income_rank", "employment_rank", "education_rank", "health_rank",
    "crime_rank", "housing_barriers_rank", "living_env_rank", "pop_density",
    "address_withheld", "name_count",
]
FEATURES = CATEGORICAL + NUMERIC

LEGAL_WORDS = r"\b(LTD|LIMITED|PLC|LLP|UK|THE)\b"


def normalise_name(names):
    """Uppercase, '&' -> AND, drop punctuation and legal words, squash spaces."""
    return (
        names.fillna("").str.upper()
        .str.replace("'", "", regex=False)
        .str.replace("&", " AND ", regex=False)
        .str.replace(r"[^A-Z0-9 ]", " ", regex=True)
        .str.replace(LEGAL_WORDS, " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def latest_enriched_file():
    files = sorted(PROCESSED_DIR.glob("london_enriched_*.csv"))
    if not files:
        raise FileNotFoundError("No enriched file found. Run src/build_dataset.py first.")
    return files[-1]


def add_features(df):
    """Engineered features that use ALL businesses (rated or not) and no ratings."""
    df = df.copy()
    name_clean = normalise_name(df["BusinessName"])
    df["name_count"] = name_clean.groupby(name_clean).transform("size")
    df.loc[name_clean == "", "name_count"] = 1
    return df


def load_rated():
    """Every business with a 0-5 rating, with engineered features and the target."""
    df = add_features(pd.read_csv(latest_enriched_file(), low_memory=False))
    rated = df[df["RatingValue"].isin(RATINGS)].copy()
    rated[TARGET] = (rated["RatingValue"].astype(int) <= 2).astype(int)
    return rated.reset_index(drop=True)


def make_splits(rated, seed=42):
    """60% train, 20% validation, 20% test.

    Stratified on borough AND target, so every borough keeps the same fail rate
    in all three parts. The test set is only used once, at the very end.
    """
    strata = rated[GROUP] + "_" + rated[TARGET].astype(str)
    train, rest = train_test_split(rated, test_size=0.4, stratify=strata, random_state=seed)
    valid, test = train_test_split(
        rest, test_size=0.5, stratify=strata.loc[rest.index], random_state=seed
    )
    return train, valid, test