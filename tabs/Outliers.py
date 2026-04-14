import streamlit as st
import pandas as pd

def render(df: pd.DataFrame):
    st.markdown("### Descriptive Statistics for Dataset With Outliers")
    st.caption(f"{len(df.columns)} features, {len(df)} entries")
    st.dataframe(df.describe())

    st.markdown("### Removing Outliers")

    st.markdown("After looking through the distributions of numeric columns and realizing IQR outlier removal was far too aggressive, " \
    "I decided to use separate methods to remove outliers from some features.")

    st.markdown("* Bounded features such as **review_scores_rating** and **host_response_rate** will be ignored")
    st.markdown("""* Features indicative of property size or value, such as **accomodates** will also be ignored;
    removing extreme **price** outliers should take care of most unreasonably-sized properties anyways.""")
    st.markdown("""* **price** will have its outliers removed using percentile clipping, since a very small number of properties
    are single handedly driving the mean up into the thousands; the 75th percentile has a value of \\$204, and the mean is \\$202, 
    with a max of \\$57,066.""")
    st.markdown("""* **minimum_nights** will have any value above 31 (a month) removed as an outlier; this removes most extreme 
                outliers, and having a minimum stay above 31 days seems unusual.""")
    
    st.markdown("### Clipping Price")
    st.markdown("##### Finding a Good Percentile to Clip")
    percentiles = [round(0.95 + i * 0.01, 2) for i in range(6)]
    percentileDf = pd.DataFrame({
        'Percentile': [f"{p:.0%}" for p in percentiles],
        'Price': [f"${df['price'].quantile(min(p, 1.0)):.2f}" for p in percentiles]
    })

    st.dataframe(percentileDf)
    st.markdown("While there are a decent amount of properties in the \\$1200 to \\$2,000 range, they are still less than 1\\% " \
    "of the dataset, and since one of my goals is to find bargains, keeping properties worth over $1,000 per night isn't " \
    "a priority.")

    st.markdown("### Descriptive Statistics After Outlier Removal")
    st.caption(f"{len(df) - len(st.session_state.finalDf)} entries removed.")
    st.caption(f"{len(st.session_state.finalDf.columns)} features, {len(st.session_state.finalDf)} entries")
    st.dataframe(st.session_state.finalDf.describe())
    
    