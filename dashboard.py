import streamlit as st

from utils import loadData

import tabs.summary as summary
import tabs.missingValues as missingValues
import tabs.featureRemoval as featureRemoval
import tabs.nonNumeric as nonNumeric
import tabs.outliers as outliers
import tabs.charts as charts
import tabs.hypothesisTesting as hypothesisTesting
import tabs.unsupervisedLearning as unsupervisedLearning
import tabs.supervisedLearning as supervisedLearning

st.set_page_config(layout="wide")


st.title("Broward County Airbnb Listings Dashboard")
df = loadData('./data/listings.csv')

# create tabs
(summaryTab, 
 missingValuesTab, 
 featureRemovalTab, 
 nonNumericTab, 
 outliersTab,
 chartsTab,
 hypothesisTestingTab,
  supervisedLearningTab,
 unsupervisedLearningTab) = st.tabs(["Raw Data Summary", 
                        "Handling Missing Values", 
                        "Feature Removal",
                        "Handling Non-Numeric Columns",
                        "Handling Outliers",
                        "Charts",
                        "Hypothesis Testing",
                        "Supervised Learning",
                        "Unsupervised Learning"])

# Stylings for subheaders and tabs
st.markdown("""
<style>
    .stTabs [data-baseweb="tab"] {
        font-size: 14px;
        padding: 8px 16px;
        border-radius: 4px 4px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #DCEDC8;
        color: black;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    [data-testid="stMarkdownContainer"] h3 {
        background-color: #D0D0D0;
        padding: 8px 4px;
        margin-bottom: 16px;
        border-radius: 2px;
    }
</style>
""", unsafe_allow_html=True)

with summaryTab:
    summary.render(df)

with missingValuesTab:
    dfNoMissingValues = missingValues.render(df)

with featureRemovalTab:
    dfFinalFeatures = featureRemoval.render(dfNoMissingValues)

with nonNumericTab:
    finalFeaturesDf = nonNumeric.render(dfFinalFeatures)

with outliersTab:
    cleanedDf = outliers.render(finalFeaturesDf)

with chartsTab:
    charts.render(cleanedDf)

with hypothesisTestingTab:
    hypothesisTesting.render(cleanedDf)

with supervisedLearningTab:
    supervisedLearning.render(cleanedDf)

with unsupervisedLearningTab:
    unsupervisedLearning.render(cleanedDf)

