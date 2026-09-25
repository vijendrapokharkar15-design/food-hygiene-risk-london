"""
Food hygiene inspection priorities for London councils.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

DATA_FILE = Path(__file__).parent / "data" / "scored.parquet"

# Model options shown to the user. Performance figures come from the
# cross-validation comparison in notebook 07 (10 paired folds).
MODELS = {
    "Standard (includes area deprivation)": {
        "key": "default",
        "recall": "41%",
        "note": (
            "Finds the most failures. Concentrates inspections in deprived "
            "neighbourhoods: 44% go to the most deprived fifth of each borough, "
            "which holds 29% of failures."
        ),
    },
    "Without area deprivation": {
        "key": "nodep",
        "recall": "39%",
        "note": (
            "Finds about 6% fewer failures, but spreads inspections across "
            "neighbourhoods in line with where failures actually are."
        ),
    },
}
VIEWS = ["Awaiting inspection only", "All businesses"]

st.set_page_config(page_title="London food hygiene risk", layout="wide")


@st.cache_data
def load_data():
    return pd.read_parquet(DATA_FILE)


df = load_data()

# --- Sidebar: the three choices that drive every view ------------------------
st.sidebar.header("Settings")

boroughs = sorted(df["LocalAuthorityName"].unique())
borough = st.sidebar.selectbox(
    "Borough", boroughs,
    index=boroughs.index("Westminster") if "Westminster" in boroughs else 0,
)

model_label = st.sidebar.radio("Risk model", list(MODELS))
model = MODELS[model_label]
st.sidebar.caption(model["note"])

view = st.sidebar.radio("Show", VIEWS)

# --- Filter to the chosen borough and view, riskiest first ---------------------
key = model["key"]
in_borough = df[df["LocalAuthorityName"] == borough]
awaiting = in_borough[in_borough["RatingValue"] == "AwaitingInspection"]
shown = awaiting if view == VIEWS[0] else in_borough
shown = shown.sort_values(f"rank_{key}")

# --- Header and headline numbers -----------------------------------------------
st.title("Food hygiene inspection priorities: London")
st.caption(f"FSA data snapshot: {df['snapshot_date'].iloc[0]}  |  Borough: {borough}")

col1, col2, col3 = st.columns(3)
col1.metric("Businesses in borough", f"{len(in_borough):,}")
col2.metric("Awaiting first inspection", f"{len(awaiting):,}")
col3.metric(
    "Failures found by inspecting the top 20%", model["recall"],
    help="Share of failing businesses (rated 0-2) found when each borough inspects "
         "the riskiest 20% of its list. Random selection finds 20%. Measured by "
         "cross-validation; the locked test set gave 39% for the standard model.",
)

# Temporary list (replaced by the map and full table in the next steps)
st.subheader(f"{view}: {len(shown):,} businesses, riskiest first")
st.dataframe(
    shown[[f"rank_{key}", "BusinessName", "BusinessType", "address", "RatingValue"]].head(50),
    width="stretch", hide_index=True,
)