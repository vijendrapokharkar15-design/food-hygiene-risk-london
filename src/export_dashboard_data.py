"""
Export a slim copy of the latest scored file for the dashboard.

The full data/ folder is not committed to Git, so the deployed dashboard reads
this small Parquet file instead. It holds only public FSA data plus our scores.

Run from the project root (after train_and_score.py):
    python src/export_dashboard_data.py
"""

import pandas as pd

from enrich_postcodes import PROJECT_ROOT
from features import PROCESSED_DIR

APP_DATA = PROJECT_ROOT / "app" / "data" / "scored.parquet"
ADDRESS_PARTS = ["AddressLine1", "AddressLine2", "AddressLine3", "AddressLine4", "PostCode"]
KEEP = [
    "FHRSID", "BusinessName", "BusinessType", "address", "LocalAuthorityName",
    "latitude", "longitude", "RatingValue", "RatingDate", "is_rated",
    "rank_default", "risk_pct_default", "reasons_up_default", "reasons_down_default",
    "rank_nodep", "risk_pct_nodep", "reasons_up_nodep", "reasons_down_nodep",
]
CATEGORY_COLUMNS = ["BusinessType", "LocalAuthorityName", "RatingValue"]


def join_address(row):
    """'1 High St', NaN, 'London', 'W3 7AA' -> '1 High St, London, W3 7AA'."""
    parts = [str(v).strip() for v in row if pd.notna(v) and str(v).strip()]
    return ", ".join(parts)


def main():
    scored_file = sorted(PROCESSED_DIR.glob("scored_*.csv"))[-1]
    snapshot_date = scored_file.stem.replace("scored_", "")
    print(f"Reading {scored_file.name}")

    df = pd.read_csv(scored_file, low_memory=False, dtype={"RatingValue": str})
    df["address"] = df[[c for c in ADDRESS_PARTS if c in df.columns]].apply(join_address, axis=1)
    df = df[KEEP].copy()

    for col in CATEGORY_COLUMNS:              # repeated text -> much smaller file
        df[col] = df[col].astype("category")
    df["snapshot_date"] = snapshot_date

    APP_DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(APP_DATA, index=False)

    size_mb = APP_DATA.stat().st_size / 1_000_000
    print(f"Saved {len(df):,} businesses, {df.shape[1]} columns, {size_mb:.1f} MB")
    print(f"   -> {APP_DATA}")


if __name__ == "__main__":
    main()