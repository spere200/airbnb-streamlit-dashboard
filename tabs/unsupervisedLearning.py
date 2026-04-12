import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.cluster import KMeans

def render(df: pd.DataFrame):
    st.subheader('Using K-Means Clustering to Group Neighborhoods by Average Price')

    # Aggregate by neighbourhood and find average price for each neighborhood
    neighborhoodDf = df.groupby('neighbourhood_cleansed')[['price', 'review_scores_rating']].mean()

    # Cluster and plot
    with st.container(border=True):
        nClusters = st.number_input('Select the Number of Clusters', min_value=2, max_value=5, value=3, key='ncluster-selector')
        clusterColors = {str(i): px.colors.qualitative.Plotly[i] for i in range(nClusters)}
        kmeans = KMeans(n_clusters=nClusters, random_state=42)
        neighborhoodDf['cluster'] = kmeans.fit_predict(neighborhoodDf[['price']])
        clusterBarChart = px.bar(neighborhoodDf.reset_index(), x='neighbourhood_cleansed', y='price',
            color=neighborhoodDf['cluster'].astype(str).values,
            color_discrete_map=clusterColors)
        clusterBarChart.update_layout(showlegend=False, xaxis_title='Neighborhoods', yaxis_title='Price')
        st.plotly_chart(clusterBarChart)

    st.subheader('Using Clusters to Find Best Neighborhoods for Each Price Range')
    col1, col2 = st.columns([1, 1])

    with col1:
        with st.container(border=True):
            bestDealNeighborhoods = neighborhoodDf.groupby('cluster')['price'].idxmin()
            bestDeals = neighborhoodDf.loc[bestDealNeighborhoods].sort_values('price')
            bestDealsPlot = px.bar(bestDeals.reset_index(), x='neighbourhood_cleansed', y='price',
                color=bestDeals['cluster'].astype('str').values,
                color_discrete_map=clusterColors)
            bestDealsPlot.update_layout(showlegend=False, title='Best Neighborhoods by Price',
                                        xaxis_title='Neighborhoods', yaxis_title='Price')
            bestDealsPlot.update_traces(texttemplate='$%{y:.0f}', textposition='outside')
            st.plotly_chart(bestDealsPlot)

    with col2:
        with st.container(border=True):
            bestRatedNeighborhoods = neighborhoodDf.groupby('cluster')['review_scores_rating'].idxmax()
            bestRated = neighborhoodDf.loc[bestRatedNeighborhoods].sort_values('price')
            bestRatedPlot = px.bar(bestRated.reset_index(), x='neighbourhood_cleansed', y='review_scores_rating',
                color=bestDeals.reset_index()['cluster'].astype(str).values,
                color_discrete_map=clusterColors)
            bestRatedPlot.update_layout(showlegend=False, title='Best Rated Neighborhood by Cluster',
                                        xaxis_title='Neighborhoods', yaxis_title='Ratings')
            bestRatedPlot.update_traces(texttemplate='%{y:.2f}', textposition='outside')
            st.plotly_chart(bestRatedPlot, key='best-ratings-plot')

