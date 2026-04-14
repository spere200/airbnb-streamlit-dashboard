import streamlit as st

from tabs import OutlierCharts, Outliers, Charts, Map

def render():
    # create tabs of the EDA page
    (outlierChartsTab,
    outlierHandlingTab,
    chartsTab,
    mapTab) = st.tabs(["Outlier Charts",
                          "Outlier Removal",
                          "Finalized Data Charts",
                          "Maps"])
    
    cleanDf = st.session_state.cleanDf
    finalDf = st.session_state.finalDf
    
    with outlierChartsTab:
        OutlierCharts.render(cleanDf)

    with outlierHandlingTab:
        Outliers.render(cleanDf)

    with chartsTab:
        Charts.render(finalDf)

    with mapTab:
        Map.render(finalDf)
    