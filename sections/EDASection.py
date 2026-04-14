import streamlit as st

from tabs import OutlierCharts, Outliers, Charts

def render():
    # create tabs of the EDA page
    (outlierChartsTab,
    outlierHandlingTab,
    chartsTab) = st.tabs(["Cleaned Data Charts",
                          "Outlier Removal",
                          "Finalized Data Charts"])
    
    cleanDf = st.session_state.cleanDf
    finalDf = st.session_state.finalDf
    
    with outlierChartsTab:
        OutlierCharts.render(cleanDf)

    with outlierHandlingTab:
        Outliers.render(cleanDf)

    with chartsTab:
        Charts.render(finalDf)
    