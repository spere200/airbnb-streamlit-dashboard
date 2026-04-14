import streamlit as st
import pandas as pd
import plotly.express as px

def render(df: pd.DataFrame):
    numericCols = df.select_dtypes(include=['number']).columns

    # excluding bounded features such as ratings, host acceptance rate, etc. or features
    # which should not have outliers removed, such as host_total_listings_count, because
    # there's no reason to remove properties owned by someone who owns a lot of properties
    excludeCols = ['host_response_rate', 'host_acceptance_rate', 'host_total_listings_count',
                   'latitude', 'longitude', 'maximum_nights', 'review_scores_rating', 
                   'review_scores_accuracy', 'review_scores_checkin', 'review_scores_cleanliness',
                   'review_scores_communication', 'review_scores_location', 'review_scores_value',
                   'number_of_reviews'] 

    colsToPlot = [col for col in numericCols if col not in excludeCols]

    st.subheader("Distributions of Data With Outliers")
    with st.container(border=True):
        distSelection = st.selectbox("Select a Feature to View its Distribution", 
                                    options=colsToPlot, 
                                    index=list(colsToPlot).index('price'),
                                    key="outlier-dist-selectbox")
        nbins = min(df[distSelection].nunique(), 100)
        distPlot = px.histogram(df, x=distSelection, nbins=nbins)
        distPlot.update_traces(texttemplate='%{y:.0f}', textposition='outside')
        st.plotly_chart(distPlot, key="outlier-dist-plot")