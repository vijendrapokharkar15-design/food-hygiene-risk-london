"""
Food hygiene inspection priorities for London councils.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

DATA_FILE = Path(__file__).parent / "data" / "scored.parquet"

st.set_page_config(page_title="London food hygiene risk", layout="wide")


@st.cache_data
def load_data():
    return pd.read_parquet(DATA_FILE)


df = load_data()

st.title("Food hygiene inspection priorities: London")
st.caption(f"FSA data snapshot: {df['snapshot_date'].iloc[0]}")
st.write(f"{len(df):,} businesses loaded across {df['LocalAuthorityName'].nunique()} boroughs.")
st.dataframe(df.head(20), width="stretch")