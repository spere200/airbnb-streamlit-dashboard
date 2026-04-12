import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.cluster import KMeans

def render(df: pd.DataFrame):
    st.subheader("Using K-Means Clustering to Group Neighborhoods by Average Price")

    # Aggregate by neighbourhood and find average price for each neighborhood
    neighborhoodDf = df.groupby('neighbourhood_cleansed')[['price']].mean()

    # Cluster
    nClusters = st.number_input("Number of Clusters", min_value=2, max_value=5, value=3)
    kmeans = KMeans(n_clusters=nClusters, random_state=42)
    neighborhoodDf['cluster'] = kmeans.fit_predict(neighborhoodDf)

    clusterBarChart = px.bar(neighborhoodDf.reset_index(), x='neighbourhood_cleansed', y='price',
             color=neighborhoodDf['cluster'].astype(str).values)
    clusterBarChart.update_layout(showlegend=False)
    st.plotly_chart(clusterBarChart)

