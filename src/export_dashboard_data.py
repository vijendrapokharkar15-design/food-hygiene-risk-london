"""
Export a slim copy of the latest scored file for the dashboard.

The full data/ folder is not committed to Git, so the deployed dashboard reads
this small Parquet file instead. It holds only public FSA data plus our scores.

Privacy: the public file holds no business names, addresses or FHRS IDs.
Location is kept only as the postcode district (e.g. "E14") and coordinates
rounded to 3 decimal places (about 100 m), so no prediction can be tied to a
named business.

Run from the project root (after train_and_score.py):
    python src/export_dashboard_data.py
"""

import pandas as pd

from enrich_postcodes import PROJECT_ROOT
from features import PROCESSED_DIR

APP_DATA = PROJECT_ROOT / "app" / "data" / "scored.parquet"
KEEP = [
    "BusinessType", "postcode_district", "LocalAuthorityName",
    "latitude", "longitude", "RatingValue", "RatingDate", "is_rated",
    "rank_default", "risk_pct_default", "reasons_up_default", "reasons_down_default",
    "rank_nodep", "risk_pct_nodep", "reasons_up_nodep", "reasons_down_nodep",
]
CATEGORY_COLUMNS = ["BusinessType", "postcode_district", "LocalAuthorityName", "RatingValue"]
COORD_DECIMALS = 3                    # 3 decimal places = roughly a 100 m grid


def postcode_district(postcode):
    """'E14 7AA' -> 'E14'. Missing (withheld address) -> 'Withheld'."""
    if pd.isna(postcode) or not str(postcode).strip():
        return "Withheld"
    return str(postcode).strip().upper().split()[0]


def main():
    scored_file = sorted(PROCESSED_DIR.glob("scored_*.csv"))[-1]
    snapshot_date = scored_file.stem.replace("scored_", "")
    print(f"Reading {scored_file.name}")

    df = pd.read_csv(scored_file, low_memory=False, dtype={"RatingValue": str})
    df["postcode_district"] = df["PostCode"].map(postcode_district)
    df["latitude"] = df["latitude"].round(COORD_DECIMALS)
    df["longitude"] = df["longitude"].round(COORD_DECIMALS)
    df = df[KEEP].copy()                      # names, addresses and IDs are dropped here

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