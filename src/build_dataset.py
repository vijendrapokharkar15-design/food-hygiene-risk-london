"""
Build the enriched business table: every business in the latest FSA snapshot
plus area features (IoD2025 deprivation ranks and Census 2021 population density).

Area features come from the most detailed level available:
    1. lsoa     - the business's own LSOA (full postcode found on postcodes.io)
    2. district - average over LSOAs in its postcode district (e.g. "CR0")
    3. borough  - average over LSOAs in its local authority

Only area data is averaged, never hygiene ratings, so no rating can leak
into a business's own features.

Run from the project root (after ingest.py and enrich_postcodes.py):
    python src/build_dataset.py
"""

import numpy as np
import pandas as pd

from enrich_postcodes import (
    FULL_POSTCODE,
    LOOKUP_FILE,
    PROJECT_ROOT,
    latest_raw_file,
    normalise_postcode,
)

EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
IOD_FILE = EXTERNAL_DIR / "iod2025_domains.csv"
DENSITY_FILE = EXTERNAL_DIR / "ts006_population_density_lsoa.csv"

# Postcode district on its own, e.g. "CR0", "SW1A", "E1"
OUTCODE = r"^[A-Z]{1,2}\d[A-Z\d]?$"

# Long official column names -> short names (rank 1 = most deprived)
IOD_COLUMNS = {
    "Index of Multiple Deprivation (IMD) Rank (where 1 is most deprived)": "imd_rank",
    "Income Rank (where 1 is most deprived)": "income_rank",
    "Employment Rank (where 1 is most deprived)": "employment_rank",
    "Education, Skills and Training Rank (where 1 is most deprived)": "education_rank",
    "Health Deprivation and Disability Rank (where 1 is most deprived)": "health_rank",
    "Crime Rank (where 1 is most deprived)": "crime_rank",
    "Barriers to Housing and Services Rank (where 1 is most deprived)": "housing_barriers_rank",
    "Living Environment Rank (where 1 is most deprived)": "living_env_rank",
}
DENSITY_COLUMN = "Population Density: Persons per square kilometre; measures: Value"
AREA_FEATURES = list(IOD_COLUMNS.values()) + ["pop_density"]


def load_area_table():
    """One row per LSOA: 8 deprivation ranks + population density."""
    iod = pd.read_csv(IOD_FILE, encoding="utf-8-sig")
    iod = iod[["LSOA code (2021)", *IOD_COLUMNS]].rename(
        columns={"LSOA code (2021)": "lsoa21", **IOD_COLUMNS}
    )

    density = pd.read_csv(DENSITY_FILE, encoding="utf-8-sig")
    density = density.rename(
        columns={"geography code": "lsoa21", DENSITY_COLUMN: "pop_density"}
    )[["lsoa21", "pop_density"]]

    return iod.merge(density, on="lsoa21", how="left", validate="one_to_one")


def fill_from(df, level, key, table):
    """Fill still-missing area features by looking up `key` in a table indexed by that key."""
    missing = df["area_match_level"].isna()
    for col in AREA_FEATURES:
        df.loc[missing, col] = df.loc[missing, key].map(table[col])
    df.loc[missing & df["imd_rank"].notna(), "area_match_level"] = level


def main():
    raw_file = latest_raw_file()
    snapshot_date = raw_file.stem.replace("london_", "")
    print(f"Reading {raw_file.name}")
    df = pd.read_csv(raw_file, low_memory=False)
    n_rows = len(df)

    # --- Postcodes --------------------------------------------------------
    df["postcode_clean"] = normalise_postcode(df["PostCode"])
    has_full_postcode = df["postcode_clean"].str.match(FULL_POSTCODE, na=False)
    df["address_withheld"] = (~has_full_postcode).astype(int)

        # Postcode district. Take the part before the space in the ORIGINAL postcode
    # when it is a valid district ("W3 7" -> "W3"); otherwise use the cleaned
    # postcode ("SW1A1AA" -> "SW1A 1AA" -> "SW1A").
    first_part_raw = df["PostCode"].str.upper().str.strip().str.split().str[0]
    first_part_clean = df["postcode_clean"].str.split(" ").str[0]
    df["outcode"] = first_part_raw.where(
        first_part_raw.str.match(OUTCODE, na=False), first_part_clean
    )
    df.loc[~df["outcode"].str.match(OUTCODE, na=False), "outcode"] = np.nan

    # --- Join the postcode lookup (LSOA + backup coordinates) --------------
    lookup = pd.read_csv(LOOKUP_FILE)
    lookup = lookup.loc[lookup["found"], ["postcode", "lsoa21", "latitude", "longitude"]]
    lookup = lookup.rename(columns={"postcode": "postcode_clean",
                                    "latitude": "pc_latitude",
                                    "longitude": "pc_longitude"})
    df = df.merge(lookup, on="postcode_clean", how="left", validate="many_to_one")

    # --- Level 1: exact LSOA ------------------------------------------------
    area = load_area_table()
    df = df.merge(area, on="lsoa21", how="left", validate="many_to_one")
    df["area_match_level"] = np.where(df["imd_rank"].notna(), "lsoa", None)

    # --- Level 2: postcode district average ---------------------------------
    district_lsoas = lookup[["postcode_clean", "lsoa21"]].copy()
    district_lsoas["outcode"] = district_lsoas["postcode_clean"].str.split(" ").str[0]
    district_table = (
        district_lsoas[["outcode", "lsoa21"]].drop_duplicates()
        .merge(area, on="lsoa21")
        .groupby("outcode")[AREA_FEATURES].mean()
    )
    fill_from(df, "district", "outcode", district_table)

    # --- Level 3: borough average -------------------------------------------
    borough_table = (
        df.loc[df["lsoa21"].notna(), ["LocalAuthorityName", "lsoa21"]].drop_duplicates()
        .merge(area, on="lsoa21")
        .groupby("LocalAuthorityName")[AREA_FEATURES].mean()
    )
    fill_from(df, "borough", "LocalAuthorityName", borough_table)

    # --- Coordinates: FSA geocode first, postcode lookup as backup ----------
    df["latitude"] = df["geocode.latitude"].fillna(df["pc_latitude"])
    df["longitude"] = df["geocode.longitude"].fillna(df["pc_longitude"])
    df["coords_source"] = np.select(
        [df["geocode.latitude"].notna(), df["pc_latitude"].notna()],
        ["fsa", "postcode"],
        default="none",
    )
    df = df.drop(columns=["pc_latitude", "pc_longitude"])

    # --- Checks and save -----------------------------------------------------
    assert len(df) == n_rows, "A join created duplicate rows"

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"london_enriched_{snapshot_date}.csv"
    df.to_csv(out_path, index=False)

    print("\n===== Summary =====")
    print(f"Rows: {len(df):,} (raw file had {n_rows:,})")
    print("\nArea features matched at level:")
    print(df["area_match_level"].value_counts(dropna=False).to_string())
    print(f"\nAddress withheld:           {df['address_withheld'].sum():,}")
    print(f"Rows missing area features: {df[AREA_FEATURES].isna().any(axis=1).sum():,}")
    print("\nCoordinates source:")
    print(df["coords_source"].value_counts().to_string())
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()