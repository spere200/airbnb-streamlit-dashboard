import streamlit as st
import pandas as pd
import plotly.express as px

def render(df: pd.DataFrame):
    numericCols = df.select_dtypes(include=['number']).columns

    st.subheader("Distributions")
    with st.container(border=True):
        distSelection = st.selectbox("Select a Feature to View its Distribution", 
                                    options=numericCols, 
                                    index=list(numericCols).index('price'),
                                    key="dist-selectbox")
        nbins = min(df[distSelection].nunique(), 50)
        distPlot = px.histogram(df, x=distSelection, nbins=nbins)
        distPlot.update_traces(texttemplate='%{y:.0f}', textposition='outside')
        st.plotly_chart(distPlot)

    st.subheader("Box Plots")
    with st.container(border=True):
        boxSelection = st.selectbox("Select a Feature to View its Distribution", 
                                    options=numericCols,
                                    index=list(numericCols).index('price'),
                                    key="box-selectbox")
        boxPlot = px.box(df, x=boxSelection)
        st.plotly_chart(boxPlot)

    nonNumericCols = df.select_dtypes(exclude='number').columns
    st.subheader("Categorical Features Frequency Plots")
    with st.container(border=True):
        catSelection = st.selectbox("Select a Feature to View its Distribution", 
                                    options=nonNumericCols,
                                    index=list(nonNumericCols).index('neighbourhood_cleansed'),
                                    key="cat-selectbox")
        catPlot = px.bar(df[catSelection].value_counts().reset_index(), x=catSelection, y='count')
        barWidth = min(0.8, 0.2 * df[catSelection].nunique()) # dynamically calculate a good bar width depending on selected col
        catPlot.update_traces(width=barWidth)
        st.plotly_chart(catPlot)
