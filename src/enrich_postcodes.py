"""
Look up every full postcode in the latest FSA snapshot on postcodes.io and save
a reusable lookup table: postcode -> LSOA 2021 code, latitude, longitude.

Postcodes already in the lookup table are skipped, so re-running on a new
snapshot only looks up new postcodes.

Run from the project root:
    python src/enrich_postcodes.py
"""

import math
import time
from pathlib import Path

import pandas as pd
import requests

# Project root = the folder above src/, wherever the script is run from
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
LOOKUP_FILE = PROJECT_ROOT / "data" / "external" / "postcode_lookup.csv"

POSTCODES_API = "https://api.postcodes.io/postcodes"
BATCH_SIZE = 100          # postcodes.io maximum per bulk request
PAUSE_SECONDS = 0.2
MAX_RETRIES = 3
LOOKUP_COLUMNS = ["postcode", "lsoa21", "latitude", "longitude", "found"]

# A full UK postcode in "OUTCODE INCODE" form, e.g. "SW1A 1AA", "E1 6AN"
FULL_POSTCODE = r"^[A-Z]{1,2}\d[A-Z\d]? \d[A-Z]{2}$"


def latest_raw_file():
    """Return the newest london_YYYY-MM-DD.csv in data/raw/."""
    files = sorted(RAW_DIR.glob("london_*.csv"))
    if not files:
        raise FileNotFoundError(f"No snapshots in {RAW_DIR}. Run src/ingest.py first.")
    return files[-1]  # ISO dates sort correctly as text


def normalise_postcode(series):
    """Uppercase, remove all spaces, then put one space before the last 3 characters."""
    compact = series.str.upper().str.replace(r"\s+", "", regex=True)
    with_space = compact.str[:-3] + " " + compact.str[-3:]
    return with_space.where(compact.str.len() >= 5, compact)


def lookup_batch(postcodes):
    """Look up up to 100 postcodes in one request. Returns one row per postcode."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                POSTCODES_API, json={"postcodes": postcodes}, timeout=30
            )
            response.raise_for_status()
            break
        except requests.RequestException as err:
            if attempt == MAX_RETRIES:
                raise
            wait = attempt * 2
            print(f"  Request failed ({err}). Retrying in {wait}s...")
            time.sleep(wait)

    rows = []
    for item in response.json()["result"]:
        result = item["result"]
        if result is None:
            rows.append({"postcode": item["query"], "lsoa21": None,
                         "latitude": None, "longitude": None, "found": False})
        else:
            rows.append({"postcode": item["query"],
                         "lsoa21": result["codes"].get("lsoa21"),
                         "latitude": result["latitude"],
                         "longitude": result["longitude"],
                         "found": True})
    return rows


def main():
    raw_file = latest_raw_file()
    print(f"Reading {raw_file.name}")
    postcodes = normalise_postcode(pd.read_csv(raw_file, usecols=["PostCode"])["PostCode"])

    is_full = postcodes.str.match(FULL_POSTCODE, na=False)
    unique_postcodes = sorted(postcodes[is_full].unique())
    print(f"Businesses:             {len(postcodes):,}")
    print(f"With a full postcode:   {is_full.sum():,}")
    print(f"Unique full postcodes:  {len(unique_postcodes):,}")

    # Caching: skip postcodes we have already looked up in a previous run
    if LOOKUP_FILE.exists():
        existing = pd.read_csv(LOOKUP_FILE)
    else:
        existing = pd.DataFrame(columns=LOOKUP_COLUMNS)
    already_done = set(existing["postcode"])
    to_lookup = [p for p in unique_postcodes if p not in already_done]
    print(f"Already in lookup table: {len(unique_postcodes) - len(to_lookup):,}")
    print(f"To look up now:          {len(to_lookup):,}\n")

    new_rows = []
    total_batches = math.ceil(len(to_lookup) / BATCH_SIZE)
    for batch_number, start in enumerate(range(0, len(to_lookup), BATCH_SIZE), start=1):
        new_rows.extend(lookup_batch(to_lookup[start:start + BATCH_SIZE]))
        if batch_number % 25 == 0 or batch_number == total_batches:
            print(f"  Batch {batch_number}/{total_batches} done")
        time.sleep(PAUSE_SECONDS)

    # Combine old and new results, skipping empty tables
    frames = [f for f in (existing, pd.DataFrame(new_rows, columns=LOOKUP_COLUMNS)) if not f.empty]
    lookup = pd.concat(frames, ignore_index=True) if frames else existing

    LOOKUP_FILE.parent.mkdir(parents=True, exist_ok=True)
    lookup.to_csv(LOOKUP_FILE, index=False)

    # Summary for the postcodes in this snapshot
    current = lookup[lookup["postcode"].isin(unique_postcodes)]
    found = int(current["found"].sum())
    print("\n===== Summary =====")
    print(f"Found:       {found:,} of {len(current):,} ({found / len(current):.1%})")
    print(f"Not found:   {len(current) - found:,}")
    print("Examples not found:", current.loc[~current["found"], "postcode"].head(10).tolist())
    print(f"\nSaved to {LOOKUP_FILE}")


if __name__ == "__main__":
    main()