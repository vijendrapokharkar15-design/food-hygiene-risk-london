"""
Model definitions shared by notebooks and training scripts.

Every model is a function  fit_predict(train, test) -> risk scores for `test`,
so they can all be trained and evaluated the same way.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from features import CATEGORICAL, FEATURES, NUMERIC, TARGET

SKEWED = ["name_count", "pop_density"]                   # long right tails: log first
OTHER_NUMERIC = [c for c in NUMERIC if c not in SKEWED]
DEPRIVATION = [c for c in NUMERIC if c.endswith("_rank")]  # the 8 IoD2025 ranks

LGBM_DEFAULTS = dict(
    n_estimators=400,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)


# --- Baseline: business type fail rate ---------------------------------------
def fit_predict_type(train, test):
    type_rates = train.groupby("BusinessType")[TARGET].mean()
    return test["BusinessType"].map(type_rates).fillna(train[TARGET].mean()).to_numpy()


# --- Logistic regression -------------------------------------------------------
def make_logreg(use_deprivation=True):
    """use_deprivation=False drops the 8 deprivation ranks (see the fairness check)."""
    scaled = OTHER_NUMERIC if use_deprivation else [
        c for c in OTHER_NUMERIC if c not in DEPRIVATION
    ]
    preprocess = ColumnTransformer([
        ("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("log_scaled", make_pipeline(FunctionTransformer(np.log1p), StandardScaler()), SKEWED),
        ("scaled", StandardScaler(), scaled),
    ])
    return Pipeline([
        ("prep", preprocess),
        ("model", LogisticRegression(class_weight="balanced", max_iter=2000)),
    ])


def fit_predict_logreg(train, test, use_deprivation=True):
    model = make_logreg(use_deprivation).fit(train[FEATURES], train[TARGET])
    return model.predict_proba(test[FEATURES])[:, 1]


# --- LightGBM --------------------------------------------------------------------
def lgb_frame(data, reference):
    """Feature table for LightGBM: categories stored as pandas 'category', using the
    category list from the reference (training) data so codes match across splits."""
    X = data[FEATURES].copy()
    for col in CATEGORICAL:
        X[col] = pd.Categorical(X[col], categories=sorted(reference[col].dropna().unique()))
    return X


def fit_predict_lgbm(train, test, **params):
    model = lgb.LGBMClassifier(**{**LGBM_DEFAULTS, **params})
    model.fit(lgb_frame(train, train), train[TARGET])
    return model.predict_proba(lgb_frame(test, train))[:, 1]


MODELS = {
    "Business type only": fit_predict_type,
    "Logistic regression": fit_predict_logreg,
    "LightGBM (default)": fit_predict_lgbm,
}