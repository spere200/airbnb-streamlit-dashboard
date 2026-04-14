import streamlit as st
import pandas as pd
import plotly.express as px

from utils import getSummaryDf

def render(df: pd.DataFrame):
    st.markdown("### Features Sorted by Missing Values")
    st.caption(f"{len(df.columns)} features, {len(df)} entries")
    summaryDf = getSummaryDf(df)
    sortedDf = summaryDf.sort_values(by="Missing Value Count", ascending=False) 
    st.dataframe(sortedDf, hide_index=True)

    st.markdown("### Deciding What to Do About Missing Values")
    st.markdown("""
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
              of about 25% of the dataset.
        3. All features related to reviews have a very similar number of missing values. I have a suspicion
           that running dropna on price will fix this, but it's worth looking into afterwards regardless.
        4. **host_location** can be removed, since there is a high amount of data missing, and the values themselves are 
          inconsistent (sometimes country, sometimes state, sometimes city, etc.)
        5. **bathrooms** has a lot of missing values, **bathrooms_text** contains the same and more info with no missing values.
           **bathrooms** can be derived from **bathrooms_text** and bathrooms text can just keep the extra info (private, shared, etc.)
             
        I'll look into host response/acceptance after dealing with the above features, since the numbers are comparatively small.
    """)

    st.markdown("##### Running dropna() on 'price' and 'estimated_revenue_l365d':")
    df = df.dropna(subset=["price", "estimated_revenue_l365d"])
    summaryDf = getSummaryDf(df)
    st.caption(f"{len(df.columns)} features, {len(df)} entries")
    st.dataframe(summaryDf)
    st.caption("It looks like the problem with beds and bedrooms was moslty resolved by the dropna price, so next, I'll be going after reviews instead")

    st.markdown("##### Running dropna() on review_scores_rating:")
    df = df.dropna(subset=["review_scores_rating"])
    summaryDf = getSummaryDf(df)
    st.caption(f"{len(df.columns)} features, {len(df)} entries")
    st.dataframe(summaryDf)

    st.markdown("##### Dropping 'host_location' and running dropna on 'host_response_time', 'host_response_rate', 'host_acceptance_rate':")
    df = df.dropna(subset=['host_response_rate', 'host_acceptance_rate'])
    df = df.drop(columns=['host_location'], axis=1)
    summaryDf = getSummaryDf(df)
    st.caption(f"{len(df.columns)} features, {len(df)} entries")
    st.dataframe(summaryDf)

    st.markdown("##### After running dropna on 'beds' (1 entry), the dataset now contains no missing values:")
    df = df.dropna(subset=["beds"])
    summaryDf = getSummaryDf(df)
    st.caption(f"{len(df.columns)} features, {len(df)} entries")
    st.dataframe(summaryDf)

    st.markdown("### Preview of Dataframe After Handling Missing Values and Removing Uninteresting Features")
    st.caption(f"{len(df.columns)} features, {len(df)} entries")
    st.dataframe(df.head())

    return df