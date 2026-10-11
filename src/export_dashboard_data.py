"""
Export a slim copy of the latest scored file for the dashboard.

The full data/ folder is not committed to Git, so the deployed dashboard reads
this small Parquet file instead. It holds only public FSA data plus our scores.

Privacy: the public file cannot be used to tie a prediction to a named business.
- No business names, addresses, FHRS IDs or per-business coordinates.
- Location is kept only as the postcode district (e.g. "E14") and the average
  location of that district's businesses within the borough, for the map.
- The inspection date is kept as the year only. An exact date, together with
  business type, district and rating, would match most businesses to their
  public FSA listing.

Run from the project root (after train_and_score.py):
    python src/export_dashboard_data.py
"""

import pandas as pd

from enrich_postcodes import PROJECT_ROOT
from features import PROCESSED_DIR

APP_DATA = PROJECT_ROOT / "app" / "data" / "scored.parquet"
KEEP = [
    "BusinessType", "postcode_district", "LocalAuthorityName",
    "district_lat", "district_lon", "RatingValue", "inspected_year", "is_rated",
    "rank_default", "risk_pct_default", "reasons_up_default", "reasons_down_default",
    "rank_nodep", "risk_pct_nodep", "reasons_up_nodep", "reasons_down_nodep",
]
CATEGORY_COLUMNS = ["BusinessType", "postcode_district", "LocalAuthorityName", "RatingValue"]
WITHHELD = "Withheld"


def postcode_district(postcode):
    """'E14 7AA' -> 'E14'. Missing (withheld address) -> 'Withheld'."""
    if pd.isna(postcode) or not str(postcode).strip():
        return WITHHELD
    return str(postcode).strip().upper().split()[0]


def add_district_location(df):
    """Average location of each district's businesses within each borough.

    Every business in a district gets the same point, so the map can show
    districts but never individual premises. Withheld addresses get no point.
    """
    group = ["LocalAuthorityName", "postcode_district"]
    centre = df.groupby(group, observed=True)[["latitude", "longitude"]].transform("mean")
    withheld = df["postcode_district"] == WITHHELD
    df["district_lat"] = centre["latitude"].where(~withheld).round(4)
    df["district_lon"] = centre["longitude"].where(~withheld).round(4)
    return df


def main():
    scored_file = sorted(PROCESSED_DIR.glob("scored_*.csv"))[-1]
    snapshot_date = scored_file.stem.replace("scored_", "")
    print(f"Reading {scored_file.name}")

    df = pd.read_csv(scored_file, low_memory=False, dtype={"RatingValue": str})
    df["postcode_district"] = df["PostCode"].map(postcode_district)
    df = add_district_location(df)
    df["inspected_year"] = pd.to_datetime(df["RatingDate"], errors="coerce").dt.year.astype("Int64")
    df = df[KEEP].copy()          # names, addresses, IDs, coordinates and exact dates dropped here

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
