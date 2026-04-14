import streamlit as st
import pandas as pd

from utils import getSummaryDf

def render(df: pd.DataFrame):
    col1, col2, col3 = st.columns(3)

    st.markdown("### Removing Unexplained/Irrelevant Features")
    col1, col2, col3 = st.columns(3)

    with col1:
        colsToDrop = ["id", "scrape_id", "listing_url", "last_scraped", "source", "name", 
                    "description", "neighborhood_overview", "picture_url", "host_id",
                    "host_url", "host_name", "host_about", "host_thumbnail_url", 
                    "host_picture_url", "host_neighbourhood", 
                    "calendar_last_scraped", "host_since", "host_response_time",
                    "first_review", "last_review"]
        
        st.markdown("##### Removing Non-Numerical, Non-Categorical Features")
        with st.container(border=True, height=420):  
            st.markdown("Removing URL's, IDs, raw text, and dates.")  
            st.markdown(", ".join([f"**{col}**" for col in colsToDrop])) 

        df = df.drop(columns=colsToDrop, axis=1)


    with col2:
        unexplainedCols = ["minimum_minimum_nights", "maximum_minimum_nights", "minimum_maximum_nights", "maximum_maximum_nights",
                           "minimum_nights_avg_ntm", "maximum_nights_avg_ntm", "calculated_host_listings_count_entire_homes",
                           "calculated_host_listings_count_private_rooms", "calculated_host_listings_count_shared_rooms", 
                           "calculated_host_listings_count"]

        st.markdown("##### Removing Unexplained Features")  
        with st.container(border=True, height=420):
            st.markdown("The following features are not explained anywhere in the dataset source, and their meaning cannot " \
            "be extrapolated from their name. As such, I am opting for their removal.")
            st.markdown(", ".join([f"**{col}**" for col in unexplainedCols]))

        df = df.drop(columns=unexplainedCols, axis=1)

    with col3:
        irrelevantCols = ['host_verifications', 'number_of_reviews_ltm', 'number_of_reviews_l30d', 'number_of_reviews_ly',
                          'reviews_per_month', 'availability_30', 'availability_60', 'availability_90', 'availability_eoy', 
                          'host_listings_count', 'host_has_profile_pic', 'number_of_reviews_ly', 'maximum_nights']
        
        st.markdown("##### Removing Irrelevant Features") 
        with st.container(border=True, height=420):
            st.markdown('**host_verifications** contains no empty rows since verification appears to be mandatory, and ' \
            '**host_identity_verified** contains much more important information than just verification method.')
            st.markdown("Since this analysis focuses on general information about the properties, granular data such as " \
            "**number_of_reviews_l30d** can be removed when more general information such as **number_of_reviews** exists.")
            st.markdown("Some columns (such as **host_has_profile_pic**) are of no interest.")
            st.markdown(", ".join([f"**{col}**" for col in irrelevantCols])) 

        df = df.drop(columns=irrelevantCols, axis=1)

    with col1:
        st.markdown("###### Removing Features With No Variance")   
        with st.container(border=True, height=180):
            noVarianceCols = [col for col in df.columns if df[col].nunique() == 1]
            st.markdown("The following features only have a single value across all rows:")
            st.markdown(", ".join([f"**{col}**" for col in noVarianceCols]))

        df = df.drop(columns=noVarianceCols, axis=1)

    with col2:
        st.markdown("###### Removing Features With a High Percentage of Missing Values")
        with st.container(border=True, height=180):
            noVarianceFeatures = []
            summaryDf = getSummaryDf(df)
            for row in summaryDf.itertuples():
                if row[-1] > len(df) * 0.9: # drop columns which have more than 90% of their values missing
                    noVarianceFeatures.append(row[1])

            st.markdown(f"The following features have a high percent of missing values:")
            st.markdown(", ".join([f"**{col}**" for col in noVarianceFeatures]))

        df = df.drop(columns=noVarianceFeatures, axis=1)

    
    with col3:
        st.markdown("###### Features to dropNa Due to Low Percentage Missing Values")
        with st.container(border=True, height=180):
            featuresToDropNa = []
            for row in summaryDf.itertuples():
                if row[-1] < len(df) * 0.05 and row[-1] > 0: # if a column has less than 5% of its rows as missing, remove those rows
                    featuresToDropNa.append(row[1])
                    
            st.markdown(f"The following relevant features have less than 5% of their values missing; removing rows with missing values:")
            st.markdown(", ".join([f"**{col}**" for col in featuresToDropNa]))

        df = df.dropna(subset=featuresToDropNa)

    st.markdown("### Feature Summary After Removal")
    st.caption(f"{len(df.columns)} features, {len(df)} entries")
    summaryDf = getSummaryDf(df)
    st.dataframe(summaryDf)

    return df
