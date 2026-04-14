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
        catCol, contCol = st.columns([1, 1])

        with catCol.container(border=True, height=760):
            # show the 3 most important ones next, then just append the rest programatically
            categoricalCols = ['room_type', 'private_bathroom']
            for c in df.select_dtypes(include=['object', 'bool']).columns.tolist():
                if c not in categoricalCols and c != 'neighbourhood_cleansed': # separate plot for neighbourhood
                    categoricalCols.append(c)

            catSelection = st.selectbox(
                "Select a Categorical Feature",
                options=categoricalCols,
                key="price-cat-selectbox"
            )
            avgPriceByCat = df.groupby(catSelection)['price'].mean().reset_index(name='avg_price')
            avgPriceByCat = avgPriceByCat.sort_values('avg_price', ascending=True)
            catPricePlot = px.bar(
                avgPriceByCat,
                title= f"Average price per {catSelection}",
                x=catSelection,
                y='avg_price',
                labels={catSelection: catSelection.replace('_', ' ').title(), 'avg_price': 'Average Price ($)'}
            )
            catPricePlot.update_traces(texttemplate='$%{y:.2f}', textposition='outside')
            st.plotly_chart(catPricePlot, use_container_width=True)

            st.markdown('---')
            st.markdown("* **room_type** and **private_bathroom** appear to be great indicators of price, " \
            "which makes sense, since both are features indicative of available space and privacy.")

        with contCol.container(border=True):
            continuousCols = ['accommodates', 'bedrooms', 'beds', 'bathrooms']

            for c in df.select_dtypes(include=['number']).columns.tolist():
                if c not in continuousCols and c not in excludeCols and c != 'price':
                    continuousCols.append(c)

            contSelection = st.selectbox("Select a Continuous Feature to Plot Against Price",
                options=continuousCols,
                key="price-cont-selectbox"
            )
            scatterPlot = px.scatter(
                df,
                title=f"{contSelection} vs price",
                x=contSelection,
                y='price',
                labels={contSelection: contSelection.replace('_', ' ').title(), 'price': 'Price ($)'},
                opacity=0.5
            )
            st.plotly_chart(scatterPlot, use_container_width=True)

            st.markdown('---')
            st.markdown("""
                        * **accomodates**, **bedrooms**, **beds**, and **bathrooms** all show a positive correlation with price
                        * **minimum_nights** shows a negative correlation with price
                        * **estimated_occupancy** shows a slight negative correlation with price, with most highly occupied properties
                          clustering in a lower price range
                        """)

        with st.container(border=True):
            avgPriceByNeighborhood = df.groupby('neighbourhood_cleansed')['price'].mean().reset_index(name='avg_price')
            avgPriceByNeighborhood = avgPriceByNeighborhood.sort_values('avg_price', ascending=True)
            neighborhoodPricePlot = px.bar(
                avgPriceByNeighborhood,
                title='Average Price by Neighborhood',
                x = 'neighbourhood_cleansed',
                y = 'avg_price',
                labels={'neighbourhood_cleansed': 'neighborhood', 'avg_price': 'Average Price ($)'},
                height=500
            )
            neighborhoodPricePlot.update_traces(texttemplate='$%{y:.2f}', textposition='outside')
            st.plotly_chart(neighborhoodPricePlot, use_container_width=True)

            st.markdown('---')
            st.markdown("""
                        While many neighborhoods cluster in a similar price range, there is a meaningful difference from 
                        \\$65 to \\$293 across the county. neighborhoods at the extremes (such as Pembroke Park and Plantation) 
                        show very distinct averages that could prove useful for a ML model. However, the predictive power 
                        of neighborhood may be limited for the majority of mid-range areas.
                        """)