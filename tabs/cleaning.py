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
                    "host_picture_url", "neighbourhood", "calendar_last_scraped"]
        
        st.write("##### Removing Irrelevant Features")
        with st.container(border=True, height=400):    
            st.caption("removing non-numerical, non-categorical, non-date/time features")
            st.write(colsToDrop)

        cleanedDf = df.drop(columns=colsToDrop, axis=1)


    with col2:
        st.write("##### Removing Features With a High Percentage of Missing Values")
        with st.container(border=True, height=400):
            st.caption("The following columns are highly unpopulated (>90\%). They can be removed:")

            featuresToBeRemoved = []
            summaryDf = getSummaryDf(cleanedDf)
            for row in summaryDf.itertuples():
                if row[-1] > len(df) * 0.9: # drop columns which have more than 90% of their values missing
                    featuresToBeRemoved.append(row[1])
                
            st.write(featuresToBeRemoved)

        cleanedDf = cleanedDf.drop(columns=featuresToBeRemoved, axis=1)

    
    with col3:
        st.write("##### Features to dropNa Due to Low Percentage Missing Values")
        with st.container(border=True, height=400):
            st.caption(f"The following columns have less than 5% of their values missing; removing rows with missing values:")

            featuresToDropNa = []
            # summaryDf = getSummaryDf(cleanedDf) technically commenting this out should never cause any problems
            for row in summaryDf.itertuples():
                if row[-1] < len(df) * 0.05: # if a column has less than 5% of its rows as missing, remove those rows
                    featuresToDropNa.append(row[1])

            st.write(featuresToDropNa)

        cleanedDf = cleanedDf.dropna(subset=featuresToDropNa)

    st.write("#### Features Summary After Removal") 
    st.caption(f"{len(cleanedDf.columns)} features, {len(cleanedDf)} entries")
    summaryDf = getSummaryDf(cleanedDf)
    st.dataframe(summaryDf)
    
    

    # st.write("#### Summary After Removal")
    # dfCleaned = df.drop(featuresToBeRemoved, axis=1)
    # dfCleaned = dfCleaned.dropna(subset=featuresToDropNa)
    # dfCleaned.reset_index(0) 
    # newSummaryDf = getSummaryDf(dfCleaned).sort_values("Missing Value Count", ascending=False)
    # st.caption(f"{len(dfCleaned.columns)} features, {len(dfCleaned)} entries")
    # st.dataframe(newSummaryDf, hide_index=True)

