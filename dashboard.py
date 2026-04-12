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
 unsupervisedLearningTab,
 supervisedLearningTab) = st.tabs(["Raw Data Summary", 
                        "Handling Missing Values", 
                        "Feature Removal",
                        "Handling Non-Numeric Columns",
                        "Removing Outliers",
                        "Charts",
                        "Hypothesis Testing",
                        "Unsupervised Learning",
                        "Supervised Learning"])

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

with unsupervisedLearningTab:
    unsupervisedLearning.render(cleanedDf)

with supervisedLearningTab:
    supervisedLearning.render(cleanedDf)


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