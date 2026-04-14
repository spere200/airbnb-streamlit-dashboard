import streamlit as st
import pandas as pd
import plotly.express as px
import json

def render(df: pd.DataFrame):
    with open('data/neighbourhoods.geojson', 'r') as f:
        geojson = json.load(f)

    # aggregate neighborhoods and keep count and average price
    aggregatedNeighborhoods = df.groupby('neighbourhood_cleansed').agg(
        count=('price', 'size'),
        avgPrice=('price', 'mean')
    ).reset_index()

    countCol, priceCol = st.columns([1, 1])

    with countCol:
        mapCountPlot = px.choropleth_mapbox(
            aggregatedNeighborhoods,
            geojson=geojson,
            locations='neighbourhood_cleansed',
            featureidkey='properties.neighbourhood',
            color='count',
            zoom=9.4,
            center={"lat": 26.145, "lon": -80.25},
            mapbox_style='carto-positron',
            opacity=0.5,
            color_continuous_scale='Reds'
        )

        mapCountPlot.update_layout(height=600, 
                                    title=dict(
                                        text='Listings by Neighbourhood',
                                        font=dict(size=28, color='#2d2d2d'),
                                        xanchor='left'
        ))
        st.plotly_chart(mapCountPlot, use_container_width=False, 
                        config={'scrollZoom': True}, key="count-map-plot")

    with priceCol:
        mapPricePlot = px.choropleth_mapbox(
            aggregatedNeighborhoods,
            geojson=geojson,
            locations='neighbourhood_cleansed',
            featureidkey='properties.neighbourhood',
            color='avgPrice',
            zoom=9.4,
            center={"lat": 26.145, "lon": -80.25},
            mapbox_style='carto-positron',
            opacity=0.5,
            color_continuous_scale='Greens'
        )

        mapPricePlot.update_layout(height=600, 
                                    title=dict(
                                        text='Neighborhood Average Price',
                                        font=dict(size=28, color='#2d2d2d'),
                                        xanchor='left'
        ))
        st.plotly_chart(mapPricePlot, use_container_width=False, 
                        config={'scrollZoom': True}, key="price-map-plot")