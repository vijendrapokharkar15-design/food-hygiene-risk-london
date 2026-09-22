"""
Download every food business in London from the FSA Food Hygiene Rating Scheme API
and save a raw, date-stamped snapshot to data/raw/.

Run from the project root:
    python src/ingest.py
"""

import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.ratings.food.gov.uk"
HEADERS = {"x-api-version": "2", "Accept": "application/json"}
PAGE_SIZE = 1000        # businesses per request
PAUSE_SECONDS = 0.5     # be polite to a public API
MAX_RETRIES = 3
RAW_DIR = Path("data/raw")


def get_json(endpoint, params=None):
    """Make one GET request. Retry with a growing wait if it fails."""
    url = f"{BASE_URL}/{endpoint}"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as err:
            if attempt == MAX_RETRIES:
                raise
            wait = attempt * 2
            print(f"  Request failed ({err}). Retrying in {wait}s...")
            time.sleep(wait)


def get_london_authorities():
    """Return ID, name and expected business count for each London authority."""
    data = get_json("Authorities")
    authorities = pd.DataFrame(data["authorities"])
    london = authorities.loc[
        authorities["RegionName"] == "London",
        ["LocalAuthorityId", "Name", "EstablishmentCount"],
    ].reset_index(drop=True)

    print(f"Found {len(london)} London authorities (expected 33).")
    return london


def get_establishments(authority_id):
    """Return every business for one authority, page by page."""
    all_rows = []
    page_number = 1
    while True:
        params = {
            "localAuthorityId": authority_id,
            "pageNumber": page_number,
            "pageSize": PAGE_SIZE,
        }
        data = get_json("Establishments", params)
        all_rows.extend(data["establishments"])

        if page_number >= data["meta"]["totalPages"]:
            break
        page_number += 1
        time.sleep(PAUSE_SECONDS)
    return all_rows


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    london = get_london_authorities()

    all_rows = []
    print(f"\n{'Authority':<30}{'Downloaded':>12}{'Expected':>10}")
    for authority in london.itertuples():
        rows = get_establishments(authority.LocalAuthorityId)
        all_rows.extend(rows)
        flag = "" if len(rows) == authority.EstablishmentCount else "  <-- mismatch"
        print(f"{authority.Name:<30}{len(rows):>12}{authority.EstablishmentCount:>10}{flag}")
        time.sleep(PAUSE_SECONDS)

    # Turn nested fields (scores, geocode) into their own columns
    df = pd.json_normalize(all_rows)
    df = df.drop(columns=["links"], errors="ignore")

    out_path = RAW_DIR / f"london_{date.today().isoformat()}.csv"
    df.to_csv(out_path, index=False)

    print("\n===== Summary =====")
    print(f"Total rows:       {len(df):,}")
    print(f"Expected:         {london['EstablishmentCount'].sum():,}")
    print(f"Unique FHRSIDs:   {df['FHRSID'].nunique():,}")
    print(f"Columns:          {df.shape[1]}")
    print("\nRatingValue counts:")
    print(df["RatingValue"].value_counts(dropna=False).to_string())
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()