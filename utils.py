import streamlit as st
import pandas as pd

# since streamlit aggressively re-renders and the dataframe is large, i'm caching loaded data for efficiency
@st.cache_data
def loadData(fileName):
    return pd.read_csv(fileName, dtype={"id":str, "scrape_id": str, "host_id": str}, low_memory=False)

# @st.cache_data
def getSummaryDf(df: pd.DataFrame):
    dataTypes = df.dtypes # data type series
    missingData = df.isna().sum() # missing data series

    # all of these have the column names as the index so they can be merged easily using pd.concat
    summaryDf = pd.concat([dataTypes, missingData], axis=1)
    summaryDf = summaryDf.reset_index(names="Feature")
    summaryDf.columns = ["Feature", "Data Type", "Missing Value Count"]
    
    return summaryDf