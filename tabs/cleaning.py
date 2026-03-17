import streamlit as st
import pandas as pd

from tabs.summary import getSummaryDf

@st.cache_data
def cleanData(df: pd.DataFrame):
    featuresToBeRemoved = []

    st.write("#### Removing Highly Unpopulated Columns") 

    for row in df.itertuples():
        if row[-1] > 90:
            featuresToBeRemoved.append(row[1])

    st.write(f"[**{", ".join(featuresToBeRemoved)}**] are highly unpopulated. They can be removed.")


def render(df: pd.DataFrame):
    st.subheader("Handling Missing Values")

    st.write("#### Features Sorted by Missing Values") 
    summaryDf = getSummaryDf(df)
    summaryDf["Missing Value %"] = (summaryDf["Missing Value Count"] / len(df) * 100).round(2)
    sortedDf = summaryDf.sort_values(by="Missing Value %", ascending=False) 
    st.dataframe(sortedDf, hide_index=True)

    cleanData(sortedDf)
    
    st.divider()