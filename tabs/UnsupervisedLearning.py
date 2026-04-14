import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def render(df: pd.DataFrame):
    st.subheader('Using K-Means Clustering to Group Neighborhoods by Average Price')

    nhoodClusterTab, dealsTab = st.tabs(["Neighborhood Price Clusters",
                                         "Finding Best and Worst Neighborhoods"])


    # Aggregate by neighbourhood and find average price for each neighborhood
    neighborhoodDf = df.groupby('neighbourhood_cleansed')[['price', 'review_scores_rating']].mean()

    # Cluster and plot
    with nhoodClusterTab:
        with st.container(border=True):
            nhoodClusters = st.number_input('Select the Number of Clusters', min_value=2, max_value=10, value=3, key='nhood-cluster-selector')
            clusterColors = {str(i): px.colors.qualitative.Plotly[i] for i in range(nhoodClusters)}
            kmeans = KMeans(n_clusters=nhoodClusters, random_state=42)
            neighborhoodDf['cluster'] = kmeans.fit_predict(neighborhoodDf[['price']])
            clusterBarChart = px.bar(neighborhoodDf.reset_index(), 
                                     title="Neighborhood Price Clusters",
                                     x='neighbourhood_cleansed', 
                                     y='price',
                                     color=neighborhoodDf['cluster'].astype(str).values,
                                     color_discrete_map=clusterColors)
            clusterBarChart.update_layout(showlegend=False, xaxis_title='Neighborhoods', yaxis_title='Price')
            st.plotly_chart(clusterBarChart)

    with dealsTab:
        priceRatingScatterDf = df.groupby('neighbourhood_cleansed').agg(
                price=('price', 'mean'),
                review_scores_rating=('review_scores_rating', 'mean')).reset_index()


        with st.container(border=True):
            priceRatingNClusters = st.number_input('Select the Number of Clusters', 
                                                   min_value=2, max_value=10, 
                                                   value=6, key='price-rating-cluster-selector')
            priceRatingScatterDf = df.groupby('neighbourhood_cleansed').agg(
                price=('price', 'mean'),
                review_scores_rating=('review_scores_rating', 'mean')
            ).reset_index()

            scaler = StandardScaler()
            scaledData = scaler.fit_transform(priceRatingScatterDf[['price', 'review_scores_rating']])

            kmeans = KMeans(n_clusters=priceRatingNClusters, random_state=42)
            priceRatingScatterDf['cluster'] = kmeans.fit_predict(scaledData).astype(str)

            scaledScatterPlot = px.scatter(
                priceRatingScatterDf,
                x='review_scores_rating',
                y='price',
                color='cluster',
                hover_data=['neighbourhood_cleansed'],
                title='Clusters of Average Price vs Average Rating by Neighbourhood',
                labels={'price': 'Average Price ($)', 'review_scores_rating': 'Average Rating'}
            )
            scaledScatterPlot.update_traces(marker=dict(size=16))
            scaledScatterPlot.update_layout(height=600, showlegend=False, yaxis_range=[0, 300])
            st.plotly_chart(scaledScatterPlot, key="scaled-scatter-plot")
