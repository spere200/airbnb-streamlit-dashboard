import streamlit as st
import pandas as pd

from tabs.summary import getSummaryDf

def render(df: pd.DataFrame):
    st.subheader("Handling Missing Values")
    
    st.write("#### Features Sorted by Missing Values") 
    st.caption(f"{len(df.columns)} features, {len(df)} entries")
    summaryDf = getSummaryDf(df)
    sortedDf = summaryDf.sort_values(by="Missing Value Count", ascending=False) 
    st.dataframe(sortedDf, hide_index=True)

    st.write("#### Removing Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        colsToDrop = ["id", "listing_url", "scrape_id", "last_scraped", "source", "name", 
                    "description", "neighborhood_overview", "picture_url", "host_id",
                    "host_url", "host_name", "host_about", "host_thumbnail_url", 
                    "host_picture_url", "host_neighbourhood", "neighbourhood", 
                    "calendar_last_scraped"]
        
        st.write("##### Removing Irrelevant Features")
        with st.container(border=True, height=400):    
            st.caption("removing non-numerical, non-categorical, non-date/time features")
            st.write(colsToDrop)

        # TODO: Find a way to cache these results to prevent needless recalculation on a re-render
        cleanedDf = df.drop(columns=colsToDrop, axis=1)


    with col2:
        st.write("##### Removing Features With a High Percentage of Missing Values")
        with st.container(border=True, height=400):
            featuresToBeRemoved = []
            summaryDf = getSummaryDf(cleanedDf)
            for row in summaryDf.itertuples():
                if row[-1] > len(df) * 0.9: # drop columns which have more than 90% of their values missing
                    featuresToBeRemoved.append(row[1])
            
            st.caption("The following columns are highly unpopulated (>90\\%). They can be removed:")
            st.write(featuresToBeRemoved)

        # TODO: Find a way to cache these results to prevent needless recalculation on a re-render
        cleanedDf = cleanedDf.drop(columns=featuresToBeRemoved, axis=1)

    
    with col3:
        st.write("##### Features to dropNa Due to Low Percentage Missing Values")
        with st.container(border=True, height=400):
            featuresToDropNa = []
            # summaryDf = getSummaryDf(cleanedDf) technically commenting this out should never cause any problems
            for row in summaryDf.itertuples():
                if row[-1] < len(df) * 0.05: # if a column has less than 5% of its rows as missing, remove those rows
                    featuresToDropNa.append(row[1])
            st.caption(f"The following columns have less than 5% of their values missing; removing rows with missing values:")
            st.write(featuresToDropNa)

        # TODO: Find a way to cache these results to prevent needless recalculation on a re-render
        cleanedDf = cleanedDf.dropna(subset=featuresToDropNa)

    st.write("#### Features Summary After Removal") 
    st.caption(f"{len(cleanedDf.columns)} features, {len(cleanedDf)} entries")
    summaryDf = getSummaryDf(cleanedDf)
    st.dataframe(summaryDf)
    
    st.write("#### Deciding What to Do About Remaining Missing Values")
    st.write("""
        Looking through the data after the initial cleaning step, the following can be performed in order of importance
        to deal with the remaining missing values:  
        1. **price** and **estimated_revenue_l365d** have the same number of missing values, so most likely entries that lack
          one lack the other, and these two features are pretty important to any sort of analysis, so entries which don't
          have them are irrelevant.
        2. **bedrooms** has no missing values, but **beds** has a lot. There are 3 options to deal with this:
            - Fill the missing values of **beds** with the corresponding value of **bedrooms**. Makes sense, but can lead to
              incorrect assumptions, since the data would not accurately show how the bed to bedroom ratio impacts other values.
            - Drop **beds** and keep only **bedrooms**. No entries are lost, but the bed to bedroom ratio is completely lost.
            - Drop all entries with a missing **beds** value. Solves missing data, keeps bedroom to bed ratio, but at the cost 
              of about 25 percent of the dataset.
        3. All features related to reviews have a very similar number of missing values, either 2,532 or 2,533. I have a suspicion
           that running dropna on price will fix this, but it's worth looking into afterwards regardless.
        4. **host_location** can be removed, since there is a high amount of data missing, and the values themselves are 
          inconsistent (sometimes country, sometimes state, sometimes city, etc.)
        5. **bathrooms** has a lot of missing values, **bathrooms_text** contains the same and more info with no missing values.
           **bathrooms** can be derived from **bathrooms_text** and bathrooms text can just keep the extra info (private, shared, etc.)
             
        I'll look into host response/acceptance after dealing with the above features, since the numbers are comparatively small.
    """)

    st.write("#### Running dropna() on 'price' and 'estimated_revenue_l365d':")
    cleanedDf = cleanedDf.dropna(subset=["price", "estimated_revenue_l365d"])
    summaryDf = getSummaryDf(cleanedDf)
    st.caption(f"{len(cleanedDf.columns)} features, {len(cleanedDf)} entries")
    st.dataframe(summaryDf)
    st.caption("It looks like the problem with beds and bedrooms was moslty resolved by the dropna price, so next, I'll be going after reviews instead")

    st.write("#### Running dropna() on review_scores_rating:")
    cleanedDf = cleanedDf.dropna(subset=["review_scores_rating"])
    summaryDf = getSummaryDf(cleanedDf)
    st.caption(f"{len(cleanedDf.columns)} features, {len(cleanedDf)} entries")
    st.dataframe(summaryDf)

    st.write("#### Dropping 'host_location' and running dropna on 'host_response_time', 'host_response_rate', 'host_acceptance_rate':")
    cleanedDf = cleanedDf.dropna(subset=['host_response_time', 'host_response_rate', 'host_acceptance_rate'])
    cleanedDf = cleanedDf.drop(columns=['host_location'], axis=1)
    summaryDf = getSummaryDf(cleanedDf)
    st.caption(f"{len(cleanedDf.columns)} features, {len(cleanedDf)} entries")
    st.dataframe(summaryDf)

    st.write("#### Running dropna on beds (1 entry), and converting bahtrooms_text to strictly categorical data")
