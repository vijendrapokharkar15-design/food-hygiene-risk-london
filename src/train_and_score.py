"""
Train the two final models on ALL rated businesses and score EVERY business
(including those awaiting inspection). Saves the models and one scored file
that the dashboard reads.

    default : logistic regression with deprivation features (chosen model)
    nodep   : the same model without the 8 deprivation ranks (fairness alternative)

Run from the project root (after build_dataset.py):
    python src/train_and_score.py
"""

import joblib
import numpy as np
import pandas as pd

from features import (
    CATEGORICAL, FEATURES, GROUP, PROCESSED_DIR, RATINGS, TARGET,
    add_features, latest_enriched_file,
)
from models import DEPRIVATION, SKEWED, make_logreg
from enrich_postcodes import PROJECT_ROOT

MODELS_DIR = PROJECT_ROOT / "models"
VARIANTS = {"default": True, "nodep": False}      # name -> use_deprivation
MIN_REASON = 0.05                                 # ignore tiny contributions

# Columns the dashboard shows (kept if present)
DISPLAY = [
    "FHRSID", "BusinessName", "BusinessType", "AddressLine1", "AddressLine2",
    "AddressLine3", "AddressLine4", "PostCode", "LocalAuthorityName",
    "latitude", "longitude", "RatingValue", "RatingDate", "address_withheld",
    "name_count",
]


def contributions(model, X):
    """Per business, how much each group of features pushes the risk score up (+) or down (-).

    Contributions are on the log-odds scale and measured against an average business:
    numeric features are standardised (0 = average), and each category is compared with
    the average category of its column.
    """
    prep = model.named_steps["prep"]
    coefs = model.named_steps["model"].coef_[0]
    Z = prep.transform(X)
    Z = Z.toarray() if hasattr(Z, "toarray") else Z

    onehot = prep.named_transformers_["categories"].get_feature_names_out(CATEGORICAL)
    scaled_cols = list(prep.transformers_[2][2])
    names = list(onehot) + SKEWED + scaled_cols

    coef = pd.Series(coefs, index=names)
    for col in CATEGORICAL:                       # centre each category block
        block = [n for n in onehot if n.startswith(col + "_")]
        coef[block] -= coef[block].mean()

    parts = pd.DataFrame(Z * coef.to_numpy(), columns=names, index=X.index)
    groups = pd.DataFrame(index=X.index)
    groups["business_type"] = parts[[n for n in onehot if n.startswith("BusinessType_")]].sum(axis=1)
    groups["chain"] = parts["name_count"]
    groups["withheld"] = parts["address_withheld"]
    groups["density"] = parts["pop_density"]
    deprivation_cols = [c for c in DEPRIVATION if c in names]
    if deprivation_cols:
        groups["deprivation"] = parts[deprivation_cols].sum(axis=1)
    return groups


def reason_text(groups, business_types):
    """Turn contributions into short readable reasons: risk-raising first."""
    labels = {
        "business_type": lambda v, bt: f"Business type: {bt}",
        "chain": lambda v, bt: "Independent business" if v > 0 else "Part of a chain",
        "withheld": lambda v, bt: "Listed premises address" if v > 0 else "Home-based / withheld address",
        "density": lambda v, bt: "Busy, densely populated area" if v > 0 else "Less densely populated area",
        "deprivation": lambda v, bt: "More deprived neighbourhood" if v > 0 else "Less deprived neighbourhood",
    }
    up, down = [], []
    for idx, row in groups.iterrows():
        bt = business_types.loc[idx]
        ordered = row.sort_values(ascending=False)
        up.append("; ".join(labels[g](v, bt) for g, v in ordered.items() if v > MIN_REASON)[:200])
        down.append("; ".join(labels[g](v, bt) for g, v in ordered[::-1].items() if v < -MIN_REASON)[:200])
    return up, down


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    enriched_file = latest_enriched_file()
    snapshot_date = enriched_file.stem.replace("london_enriched_", "")
    print(f"Reading {enriched_file.name}")

    df = add_features(pd.read_csv(enriched_file, low_memory=False))
    df["is_rated"] = df["RatingValue"].isin(RATINGS)
    rated = df[df["is_rated"]].copy()
    rated[TARGET] = (rated["RatingValue"].astype(int) <= 2).astype(int)
    df.loc[rated.index, TARGET] = rated[TARGET]
    print(f"All businesses: {len(df):,}   Rated (training): {len(rated):,}   "
          f"Unrated (scored only): {(~df['is_rated']).sum():,}")

    out = df[[c for c in DISPLAY if c in df.columns] + ["is_rated", TARGET]].copy()

    for variant, use_deprivation in VARIANTS.items():
        model = make_logreg(use_deprivation).fit(rated[FEATURES], rated[TARGET])
        joblib.dump(model, MODELS_DIR / f"logreg_{variant}.joblib")

        score = model.predict_proba(df[FEATURES])[:, 1]
        out[f"risk_{variant}"] = score
        # Within-borough position: 1 = riskiest; percentile 1.0 = riskiest
        out[f"rank_{variant}"] = out.groupby(GROUP)[f"risk_{variant}"].rank(
            ascending=False, method="first").astype(int)
        out[f"risk_pct_{variant}"] = out.groupby(GROUP)[f"risk_{variant}"].rank(pct=True)

        up, down = reason_text(contributions(model, df[FEATURES]), df["BusinessType"])
        out[f"reasons_up_{variant}"] = up
        out[f"reasons_down_{variant}"] = down
        print(f"Model '{variant}' trained and saved; all businesses scored")

    assert out[[f"risk_{v}" for v in VARIANTS]].notna().all().all(), "Some businesses have no score"

    out_path = PROCESSED_DIR / f"scored_{snapshot_date}.csv"
    out.to_csv(out_path, index=False)

    # Sanity check: the 5 riskiest unrated businesses in the largest borough
    borough = out[GROUP].value_counts().index[0]
    sample = out[(out[GROUP] == borough) & ~out["is_rated"]].nsmallest(5, "rank_default")
    print(f"\nTop 5 unrated businesses in {borough} (default model):")
    print(sample[["BusinessName", "BusinessType", "risk_default", "rank_default",
                  "reasons_up_default"]].to_string(index=False))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()