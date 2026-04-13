import streamlit as st
import pandas as pd

from tabs import Summary, MissingValues, FeatureRemoval, NonNumeric, Outliers

def render(df: pd.DataFrame):
    # create tabs of the data cleaning page
    (summaryTab,
    missingValuesTab,
    featureRemovalTab,
    nonNumericTab,
    outliersTab) = st.tabs(["Raw Data Summary",
                            "Handling Missing Values",
                            "Feature Removal",
                            "Converting Non-Numeric Features",
                            "Outlier Removal"])
    
    with summaryTab:
        Summary.render(df)

    with missingValuesTab:
        dfNoMissing = MissingValues.render(df)

    with featureRemovalTab:
        dfFinalFeatures = FeatureRemoval.render(dfNoMissing)

    with nonNumericTab:
        dfConvertedFeatures = NonNumeric.render(dfFinalFeatures)

    # store the finalized DF in session state so other pages have access to it
    with outliersTab:
        st.session_state.cleanDf = Outliers.render(dfConvertedFeatures)
    