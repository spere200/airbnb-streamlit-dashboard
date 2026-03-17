import streamlit as st
# import pandas as pd

from utils import loadData

import tabs.summary as summary
import tabs.cleaning as cleaning

st.set_page_config(layout="wide")


st.title("Broward County Airbnb Listings Dashboard")
df = loadData('./data/listings.csv')

# create tabs
summaryTab, cleaningTab = st.tabs(["Raw Data Summary", "Data Cleaning"])

with summaryTab:
    summary.render(df)

with cleaningTab:
    cleaning.render(df)


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