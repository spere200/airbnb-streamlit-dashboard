import streamlit as st
import pandas as pd
import plotly.express as px

def render(df: pd.DataFrame):
    numericDf = df.select_dtypes(include=['number'])
    numericColumns = numericDf.columns.tolist()

    st.subheader("Distributions")
    distSelection = st.selectbox("Select a Feature to View its Distribution", 
                                 options=numericColumns, 
                                 index=list(numericColumns).index('price'))
    nbins = min(df[distSelection].nunique(), 50)
    distPlot = px.histogram(df, x=distSelection, nbins=nbins)
    st.plotly_chart(distPlot)