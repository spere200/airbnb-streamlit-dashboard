import streamlit as st
import pandas as pd

from tabs import FeatureRemoval, MissingValues, Summary, NonNumeric, _remove_outliers, CorrelatedFeatures

def render(df: pd.DataFrame):
    # create tabs of the data cleaning page
    (summaryTab,
    featureRemovalTab,
    missingValuesTab,
    nonNumericTab,
    CorrelatedFeaturesTab) = st.tabs(["Raw Data Summary",
                            "Feature Removal",
                            "Handling Missing Values",
                            "Converting Non-Numeric Features",
                            "Checking Correlations"]) 
    
    with summaryTab:
        Summary.render(df)

    with featureRemovalTab:
        dfFinalFeatures = FeatureRemoval.render(df)

    with missingValuesTab:
        dfNoMissing = MissingValues.render(dfFinalFeatures)

    # store the finalized DF in session state so other pages have access to it
    with nonNumericTab:
        cleanDf = NonNumeric.render(dfNoMissing)
        if 'cleanDf' not in st.session_state:
            st.session_state.cleanDf = cleanDf
            cleanDf.to_csv('data/cleanedDf.csv', index=False)

        # secrely remove outliers and store the final dataframe with no outliers; done for better performance
        if 'finalDf' not in st.session_state:
            st.session_state.finalDf = _remove_outliers.removeOutliers(st.session_state.cleanDf)
            st.session_state.finalDf.to_csv('data/finalDf.csv', index=False)

    with CorrelatedFeaturesTab:
        CorrelatedFeatures.render(st.session_state.finalDf)
