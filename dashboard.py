import streamlit as st
import pandas as pd

# since streamlit aggressively re-renders and the dataframe is large, i'm caching loaded data for efficiency
@st.cache_data
def loadData():
    return pd.read_csv('./data/listings.csv', dtype={"id":str, "scrape_id": str, "host_id": str})

from tabs.summary import render

st.set_page_config(layout="wide")

st.title("Broward County Airbnb Listings Dashboard")

# Load Data
df = loadData()

summaryTab, visTab = st.tabs(["Data Summary", "Charts"])

with summaryTab:
    render(df)


# DONE Load & inspect — Get the data into memory, check shape, columns, data types
# Check for missing values — Identify gaps, decide how to handle them
# Examine distributions — Histograms, box plots for numerical features; value counts for categorical
# DONE Summarize statistics — Mean, median, std dev, quartiles, etc.
# Look for outliers — Identify unusual values that might skew analysis
# Explore relationships — Correlations between variables, scatter plots, cross-tabulations
# Check data quality — Duplicates, inconsistencies, typos, unexpected values
# Segment & group — Break data into subsets to spot patterns
# Visualize patterns — Create charts to reveal trends, clusters, anomalies
# Generate hypotheses — Note interesting findings to test later
