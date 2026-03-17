import streamlit as st
import pandas as pd

def render(df: pd.DataFrame):
    st.subheader("Dataframe Preview")
    previewVisible = st.checkbox("Hide/Show", value=False)
    if previewVisible:
        st.dataframe(df.head())
        st.divider()

    st.subheader("Details")

    st.write("##### Feature Summary")
    st.caption(f"{len(df.columns)} features, {len(df)} entries")

    dataTypes = df.dtypes # data type series
    missingData = df.isna().sum() # missing data series

    # all of these have the column names as the index so they can be merged easily using pd.concat
    summaryDf = pd.concat([dataTypes, missingData], axis=1)
    summaryDf = summaryDf.reset_index(names="Feature")
    summaryDf.columns = ["Feature", "Data Type", "Missing Value Count"]

    st.dataframe(summaryDf)

    st.write("##### Descriptive Statistics")
    st.dataframe(df.describe())