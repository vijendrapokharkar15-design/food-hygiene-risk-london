"""
Food hygiene inspection priorities for London councils.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
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
TOP_SHARE = 0.2                      # the map shows the top 20% of the current list

RATING_TEXT = {
    "AwaitingInspection": "Awaiting inspection",
    "AwaitingPublication": "Awaiting publication",
    "Exempt": "Exempt",
}

# Map dot colours: one orange hue, light (lower priority) to dark (highest)
LOW_COLOUR = np.array([246, 173, 125])
HIGH_COLOUR = np.array([176, 58, 12])

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
shown = shown.sort_values(f"rank_{key}").copy()
shown["priority"] = np.arange(1, len(shown) + 1)     # 1 = first to inspect in this list

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

# --- Map: the top 20% of the current list ---------------------------------------
top_n = int(np.ceil(TOP_SHARE * len(shown)))
top = shown.head(top_n)
mappable = top.dropna(subset=["latitude", "longitude"]).copy()

st.subheader(f"Map: top 20% of this list ({top_n:,} businesses)")
not_mapped = top_n - len(mappable)
if not_mapped:
    st.caption(f"{not_mapped:,} of these have a withheld address, so they appear in the "
               "list below but not on the map.")

if mappable.empty:
    st.info("None of these businesses have a published location.")
else:
    # 0 = lowest priority on the map, 1 = highest -> interpolate the colour
    strength = 1 - (mappable["priority"] - 1) / max(len(top) - 1, 1)
    colours = LOW_COLOUR + np.outer(strength, HIGH_COLOUR - LOW_COLOUR)
    mappable["colour"] = colours.round().astype(int).tolist()
    mappable["reasons"] = mappable[f"reasons_up_{key}"].fillna("")
    for col in ["BusinessName", "BusinessType", "address"]:
        mappable[col] = mappable[col].astype(str)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=mappable[["longitude", "latitude", "colour", "priority",
                       "BusinessName", "BusinessType", "address", "reasons"]],
        get_position=["longitude", "latitude"],
        get_fill_color="colour",
        get_radius=30,
        radius_min_pixels=4,
        radius_max_pixels=10,
        stroked=True,
        get_line_color=[255, 255, 255, 180],
        line_width_min_pixels=1,
        pickable=True,
    )
    view_state = pdk.ViewState(
        latitude=mappable["latitude"].median(),
        longitude=mappable["longitude"].median(),
        zoom=12,
    )
    tooltip = {
        "html": "<b>#{priority} {BusinessName}</b><br/>{BusinessType}<br/>{address}"
                "<br/><br/><i>Why: {reasons}</i>",
        "style": {"fontSize": "12px", "maxWidth": "320px"},
    }
    st.pydeck_chart(
        pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip, map_style=None),
        height=520,
    )
    st.caption("Darker dots = higher priority. Hover over a dot for details.")

# --- Ranked table and download ----------------------------------------------------
st.subheader(f"{view}: {len(shown):,} businesses, riskiest first")

search = st.text_input("Search by business name", placeholder="e.g. Pret, Tesco, Kebab")
listed = shown
if search:
    listed = shown[shown["BusinessName"].str.contains(search, case=False, na=False, regex=False)]


def top_percent(pct):
    """0.987 (riskier than 98.7% of the borough) -> 'Top 2%'."""
    return f"Top {max(1, math.ceil((1 - pct) * 100))}%"


table = pd.DataFrame({
    "Priority": listed["priority"],
    "Business": listed["BusinessName"].astype(str),
    "Type": listed["BusinessType"].astype(str),
    "Address": listed["address"],
    "Current rating": listed["RatingValue"].astype(str).replace(RATING_TEXT),
    "Last inspected": pd.to_datetime(listed["RatingDate"], errors="coerce")
                        .dt.strftime("%d %b %Y").where(listed["is_rated"], "").fillna(""),
    "Risk in borough": listed[f"risk_pct_{key}"].map(top_percent),
    "Why it ranks high": listed[f"reasons_up_{key}"].fillna(""),
    "Lowers its risk": listed[f"reasons_down_{key}"].fillna(""),
})

st.dataframe(table, width="stretch", height=480, hide_index=True)

st.download_button(
    "Download this list (CSV)",
    data=table.to_csv(index=False).encode("utf-8"),
    file_name=f"inspection_priorities_{borough.replace(' ', '_')}_{key}.csv",
    mime="text/csv",
)

# --- About ---------------------------------------------------------------------------
with st.expander("About this tool and its limits"):
    st.markdown(
        """
**What it does.** Ranks food businesses within each London borough by their risk of
receiving a food hygiene rating of 0, 1 or 2, so inspectors can visit the riskiest first.

**What it uses.** Only information known *before* an inspection: business type, whether it
is part of a chain, whether its address is withheld (usually home-based), population density
and (standard model only) area deprivation (English Indices of Deprivation 2025). Past
ratings and inspection scores are never used as inputs.

**How well it works.** On businesses the model never saw, inspecting the top 20% of each
borough's list found about 39-41% of failing businesses, roughly double random selection.

**Limits.**
- A ranking, not a verdict. Most high-priority businesses will pass their inspection.
- The FSA category "Retailers - other" mixes grocers and butchers with pharmacies and
  toy libraries, so some high-ranked retailers are low risk in reality.
- Councils rate with different strictness, so rankings are only meaningful within a borough.
- The standard model sends noticeably more inspections to deprived neighbourhoods than
  their share of failures. The alternative model avoids this at a small cost in accuracy.
- Ratings are recorded by human inspectors; any bias in how areas are rated is learned by
  the model.

Data: Food Standards Agency (FHRS API), postcodes.io, MHCLG IoD2025, ONS Census 2021.
"""
    )