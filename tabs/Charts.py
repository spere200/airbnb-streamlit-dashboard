import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

def render(df: pd.DataFrame):

    distTab, priceTab, = st.tabs([
        "Distributions",
        "What Drives Price?"
    ])

    with distTab:
        numericCols = df.select_dtypes(include=['number']).columns

        # excluding bounded features such as ratings, host acceptance rate, etc. or features
        # which should not have outliers removed, such as host_total_listings_count, because
        # there's no reason to remove properties owned by someone who owns a lot of properties
        excludeCols = ['host_response_rate', 'host_acceptance_rate', 'host_total_listings_count',
                    'latitude', 'longitude', 'maximum_nights', 'review_scores_rating', 
                    'review_scores_accuracy', 'review_scores_checkin', 'review_scores_cleanliness',
                    'review_scores_communication', 'review_scores_location', 'review_scores_value',
                    'number_of_reviews'] 

        numericColsToPlot = [col for col in numericCols if col not in excludeCols]

        with st.container(border=True):
            distSelection = st.selectbox("Select a Feature to View its Distribution", 
                                        options=numericColsToPlot, 
                                        index=list(numericColsToPlot).index('price'),
                                        key="dist-selectbox")
            nbins = min(df[distSelection].nunique(), 100)
            distPlot = px.histogram(df, x=distSelection, nbins=nbins)
            distPlot.update_traces(texttemplate='%{y:.0f}', textposition='outside')
            st.plotly_chart(distPlot, key="dist-plot")

    with priceTab:
        priceRoomCol, priceNeighborhoodCol = st.columns([1, 1])

        # with priceRoomCol.container(border=True):
        #     avgPriceByRoom = df.groupby('room_type')['price'].mean().reset_index(name='avg_price')
        #     avgPriceByRoom = avgPriceByRoom.sort_values('avg_price', ascending=True)
        #     roomPricePlot = px.bar(
        #         avgPriceByRoom,
        #         x = 'room_type',
        #         y = 'avg_price',
        #         labels={'room_type': 'Room Type', 'avg_price': 'Average Price ($)'}
        #     )
        #     roomPricePlot.update_traces(texttemplate='$%{y:.2f}', textposition='outside')
        #     st.plotly_chart(roomPricePlot, use_container_width=True)

        # with priceNeighborhoodCol.container(border=True):
        #     avgPriceByNeighborhood = df.groupby('neighbourhood_cleansed')['price'].mean().reset_index(name='avg_price')
        #     avgPriceByNeighborhood = avgPriceByNeighborhood.sort_values('avg_price', ascending=True)
        #     neighborhoodPricePlot = px.bar(
        #         avgPriceByNeighborhood,
        #         x = 'neighbourhood_cleansed',
        #         y = 'avg_price',
        #         labels={'neighbourhood_cleansed': 'Neighbourhood', 'avg_price': 'Average Price ($)'}
        #     )
        #     neighborhoodPricePlot.update_traces(texttemplate='$%{x:.2f}', textposition='outside')
        #     st.plotly_chart(neighborhoodPricePlot, use_container_width=True)

        with st.container(border=True):
            categoricalCols = ['room_type', 'private_bathroom', 'neighbourhood_cleansed']
            catSelection = st.selectbox(
                "Select a Categorical Feature to Plot Against Price",
                options=categoricalCols,
                key="price-cat-selectbox"
            )
            avgPriceByCat = df.groupby(catSelection)['price'].mean().reset_index(name='avg_price')
            avgPriceByCat = avgPriceByCat.sort_values('avg_price', ascending=True)
            catPricePlot = px.bar(
                avgPriceByCat,
                x=catSelection,
                y='avg_price',
                labels={catSelection: catSelection.replace('_', ' ').title(), 'avg_price': 'Average Price ($)'}
            )
            catPricePlot.update_traces(texttemplate='$%{y:.2f}', textposition='outside')
            st.plotly_chart(catPricePlot, use_container_width=True)