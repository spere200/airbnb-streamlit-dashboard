import streamlit as st
import pandas as pd

from utils import getSummaryDf

def render(df: pd.DataFrame):
    st.markdown("### Data Source")
    st.markdown("Data sourced from [Inside Airbnb](https://insideairbnb.com/get-the-data/), " \
    "licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)")
    st.markdown("### Dataframe Preview")
    st.dataframe(df.head())
    st.markdown("### Details")
    st.write("##### Feature Summary")
    st.caption(f"{len(df.columns)} features, {len(df)} entries")
    summaryDf = getSummaryDf(df)
    st.dataframe(summaryDf)
    st.markdown("### Descriptive Statistics")
    st.dataframe(df.describe())