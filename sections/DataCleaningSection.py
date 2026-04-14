import streamlit as st
import pandas as pd

from tabs import Summary, MissingValues, FeatureRemoval, NonNumeric, _remove_outliers

def render(df: pd.DataFrame):
    # create tabs of the data cleaning page
    (summaryTab,
    missingValuesTab,
    featureRemovalTab,
    nonNumericTab) = st.tabs(["Raw Data Summary",
                            "Handling Missing Values",
                            "Feature Removal",
                            "Converting Non-Numeric Features"])
    
    with summaryTab:
        Summary.render(df)

    with missingValuesTab:
        dfNoMissing = MissingValues.render(df)

    with featureRemovalTab:
        dfFinalFeatures = FeatureRemoval.render(dfNoMissing)

    # store the finalized DF in session state so other pages have access to it
    with nonNumericTab:
        cleanDf = NonNumeric.render(dfFinalFeatures)
        if 'cleanDf' not in st.session_state:
            st.session_state.cleanDf = cleanDf
            cleanDf.to_csv('data/cleanedDf.csv', index=False)

        if 'finalDf' not in st.session_state:
            st.session_state.finalDf = _remove_outliers.removeOutliers(st.session_state.cleanDf)
            st.session_state.finalDf.to_csv('data/finalDf.csv', index=False)
