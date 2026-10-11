"""
Food hygiene inspection priorities for London councils.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

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
VIEWS = ["All businesses", "Awaiting inspection only"]
MIN_GROUP = 10                       # districts and business types with fewer are hidden
FAIL_RATINGS = ["0", "1", "2"]

# Priority bands, from each business's percentile within its borough
# (1.0 = highest priority). High matches the top 20% behind the headline figure.
BANDS = ["High", "Medium", "Low"]
HIGH_CUTOFF = 0.8                    # top 20%
MEDIUM_CUTOFF = 0.5                  # next 30%; the bottom 50% is Low
MIXED_CATEGORY = "Retailers - other"

RATING_TEXT = {
    "AwaitingInspection": "Awaiting inspection",
    "AwaitingPublication": "Awaiting publication",
    "Exempt": "Exempt",
}

# Map colours: one orange hue, light (few High-priority businesses) to dark (many)
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

def priority_band(pct):
    """Within-borough percentile -> 'High' (top 20%), 'Medium' (next 30%) or 'Low'."""
    if pct >= HIGH_CUTOFF:
        return "High"
    if pct >= MEDIUM_CUTOFF:
        return "Medium"
    return "Low"


# --- Filter to the chosen borough and view, highest priority first ---------------------
key = model["key"]
in_borough = df[df["LocalAuthorityName"] == borough]
awaiting = in_borough[in_borough["RatingValue"] == "AwaitingInspection"]
shown = in_borough if view == VIEWS[0] else awaiting
shown = shown.sort_values(f"rank_{key}").copy()
shown["priority"] = np.arange(1, len(shown) + 1)     # 1 = first to inspect in this list
shown["band"] = shown[f"risk_pct_{key}"].map(priority_band)

# --- Header and headline numbers -----------------------------------------------
st.title("Food hygiene inspection priorities: London")
st.caption(f"FSA data snapshot: {df['snapshot_date'].iloc[0]}  |  Borough: {borough}")
st.info(
    "Portfolio project using public Food Standards Agency data. Scores are model "
    "estimates of inspection priority, not ratings or judgements about any business. "
    "Business names, exact locations and exact inspection dates are deliberately not "
    "shown. Not affiliated with the FSA or any local authority."
)

col1, col2, col3 = st.columns(3)
col1.metric("Businesses in borough", f"{len(in_borough):,}")
col2.metric("Awaiting first inspection", f"{len(awaiting):,}")
col3.metric(
    "Failures found by inspecting the top 20%", model["recall"],
    help="Share of failing businesses (rated 0-2) found when each borough inspects "
         "the highest-priority 20% of its list. Random selection finds 20%. Measured by "
         "cross-validation; the locked test set gave 39% for the standard model.",
)

# --- Map: postcode districts, never individual businesses ----------------------------
st.subheader("Map: High-priority share by postcode district")

districts = (
    shown[shown["postcode_district"] != "Withheld"]
    .groupby("postcode_district", observed=True)
    .agg(businesses=("band", "size"),
         high=("band", lambda b: (b == "High").sum()),
         latitude=("district_lat", "first"),
         longitude=("district_lon", "first"))
    .reset_index()
    .dropna(subset=["latitude", "longitude"])
)
hidden = (districts["businesses"] < MIN_GROUP).sum()
districts = districts[districts["businesses"] >= MIN_GROUP].copy()

if districts.empty:
    st.info(f"Every postcode district in this list has fewer than {MIN_GROUP} businesses, "
            "so none are mapped. Switch to 'All businesses' to see the map.")
else:
    districts["share"] = districts["high"] / districts["businesses"]
    districts["share_text"] = (districts["share"] * 100).round().astype(int).astype(str) + "%"
    districts["postcode_district"] = districts["postcode_district"].astype(str)
    # Darkest colour at 50% High or more (the borough average is 20%)
    strength = (districts["share"] / 0.5).clip(0, 1)
    colours = LOW_COLOUR + np.outer(strength, HIGH_COLOUR - LOW_COLOUR)
    districts["colour"] = colours.round().astype(int).tolist()
    districts["radius"] = 12 * np.sqrt(districts["businesses"])   # area grows with count

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=districts[["longitude", "latitude", "colour", "radius", "postcode_district",
                        "businesses", "high", "share_text"]],
        get_position=["longitude", "latitude"],
        get_fill_color="colour",
        get_radius="radius",
        opacity=0.6,
        stroked=True,
        get_line_color=[255, 255, 255, 180],
        line_width_min_pixels=1,
        pickable=True,
    )
    view_state = pdk.ViewState(
        latitude=districts["latitude"].median(),
        longitude=districts["longitude"].median(),
        zoom=12,
    )
    tooltip = {
        "html": "<b>{postcode_district}</b><br/>{businesses} businesses<br/>"
                "High priority: {high} ({share_text})",
        "style": {"fontSize": "12px"},
    }
    st.pydeck_chart(
        pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip, map_style=None),
        height=520,
    )
    note = (f" {hidden} district(s) with fewer than {MIN_GROUP} businesses are not shown."
            if hidden else "")
    st.caption("One circle per postcode district. Bigger = more businesses, darker = larger "
               "share in the High band (borough average 20%). Hover for details." + note)

# --- Summary by business type ----------------------------------------------------------
st.subheader(f"By business type: {borough}")

by_type = in_borough.assign(
    band=in_borough[f"risk_pct_{key}"].map(priority_band),
    failed=in_borough["RatingValue"].isin(FAIL_RATINGS),
)
summary = (
    by_type.groupby("BusinessType", observed=True)
    .agg(businesses=("band", "size"),
         high=("band", lambda b: (b == "High").sum()),
         rated=("is_rated", "sum"),
         failed=("failed", "sum"))
    .reset_index()
)
summary = summary[summary["businesses"] >= MIN_GROUP].sort_values("high", ascending=False)
summary_table = pd.DataFrame({
    "Business type": summary["BusinessType"].astype(str),
    "Businesses": summary["businesses"],
    "High priority": summary["high"],
    "Share in High band": (summary["high"] / summary["businesses"]).map("{:.0%}".format),
    "Actual failure rate (rated 0-2)": (summary["failed"] / summary["rated"].where(summary["rated"] > 0))
                                       .map(lambda v: "" if pd.isna(v) else f"{v:.0%}"),
})
st.dataframe(summary_table, width="stretch", hide_index=True)
st.caption(
    "Actual failure rate is the share of already-inspected businesses of that type rated "
    f"0, 1 or 2. Compare it with the share in the High band to see where the model "
    f"over- or under-flags a type. Types with fewer than {MIN_GROUP} businesses are not shown."
)

# --- Ranked table and download ----------------------------------------------------
listed = shown

st.subheader(f"{view}: {len(listed):,} businesses, highest priority first")

counts = listed["band"].value_counts()
st.caption(
    "  |  ".join(f"{band} priority: {counts.get(band, 0):,}" for band in BANDS)
    + "  (High = top 20% in the borough, Medium = next 30%, Low = bottom 50%)"
)

# The FSA's "Retailers - other" category mixes grocers with pharmacies and other
# low-risk shops, and it fills a large part of the High band. Say so for this borough.
high = in_borough[in_borough[f"risk_pct_{key}"] >= HIGH_CUTOFF]
mixed_share = (high["BusinessType"] == MIXED_CATEGORY).mean() if len(high) else 0
if mixed_share >= 0.1:
    st.warning(
        f"{mixed_share:.0%} of High-priority businesses in {borough} are "
        f"'{MIXED_CATEGORY}', an FSA category that mixes grocers with pharmacies and "
        "other low-risk shops. Treat these with extra caution."
    )


table = pd.DataFrame({
    "Priority": listed["priority"],
    "Type": listed["BusinessType"].astype(str),
    "Area": listed["postcode_district"].astype(str),
    "Current rating": listed["RatingValue"].astype(str).replace(RATING_TEXT),
    "Last inspected (year)": listed["inspected_year"].astype("string")
                               .where(listed["is_rated"], "").fillna(""),
    "Priority band": listed["band"],
    "Why it's prioritised": listed[f"reasons_up_{key}"].fillna(""),
    "Lowers its priority": listed[f"reasons_down_{key}"].fillna(""),
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
**What it does.** Ranks food businesses within each London borough by inspection priority:
how likely the model thinks a business is to receive a food hygiene rating of 0, 1 or 2.
The aim is to help inspectors decide where to look first.

**Priority bands.** High = top 20% of the borough, Medium = next 30%, Low = bottom 50%.
A band is a suggestion about inspection order, not a rating or judgement of the business.

**What it uses.** Only information known *before* an inspection: business type, whether it
is part of a chain, whether its address is withheld (usually home-based), population density
and (standard model only) area deprivation (English Indices of Deprivation 2025). Past
ratings and inspection scores are never used as inputs.

**How well it works.** On businesses the model never saw, inspecting the High band (top 20%)
of each borough's list found about 39-41% of failing businesses, roughly double random selection.

**Limits.**
- A ranking, not a verdict. Most high-priority businesses will pass their inspection.
- The FSA category "Retailers - other" mixes grocers and butchers with pharmacies and
  toy libraries. It makes up over half of the High band across London, so many
  High-priority retailers are low risk in reality.
- Councils rate with different strictness, so rankings are only meaningful within a borough.
- The standard model sends noticeably more inspections to deprived neighbourhoods than
  their share of failures. The alternative model avoids this at a small cost in accuracy.
- Ratings are recorded by human inspectors; any bias in how areas are rated is learned by
  the model.

Data: Food Standards Agency (FHRS API), postcodes.io, MHCLG IoD2025, ONS Census 2021.
"""
    )