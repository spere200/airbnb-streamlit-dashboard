import streamlit as st
import pandas as pd

from utils import getSummaryDf

def render(df: pd.DataFrame):
    st.subheader("Data Source")
    st.markdown("Data sourced from [Inside Airbnb](https://insideairbnb.com/get-the-data/), " \
    "licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)")

    st.subheader("Dataframe Preview")

    previewVisible = st.checkbox("Hide/Show", key=f"summaryPrevCheckbox", value=False)
    if previewVisible:
        st.dataframe(df.head())

    st.subheader("Details")

    st.write("##### Feature Summary")
    st.caption(f"{len(df.columns)} features, {len(df)} entries")

    summaryDf = getSummaryDf(df)

    st.dataframe(summaryDf)

    st.write("##### Descriptive Statistics")
    st.dataframe(df.describe())