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
    st.markdown("""* Features indicative of property size or value, such as **accomodates** were also ignored;
    removing extreme **price** outliers should take care of most unreasonably-sized properties anyways.""")
    st.markdown("""* **price** will have its outliers removed using percentile clipping, since a very small number of properties
    are single handedly driving the mean up into the thousands; the 75th percentile has a value of \\$204, and the mean is \\$202, 
    with a max of \\$57,066.""")
    st.markdown("""* **minimum_nights** will have any value above 31 (a month) removed as an outlier; this removes most extreme 
                outliers, and having a minimum stay above 30 days seems unusual.""")
    
    st.markdown("### Clipping Price")
    st.markdown("##### Finding a Good Percentile to Clip")
    percentiles = [0.95, 0.96, 0.97, 0.98, 0.99, 1]
    percentileDf = pd.DataFrame({
        'Percentile': [f"{p:.0%}" for p in percentiles],
        'Price': [f"${df['price'].quantile(p):.2f}" for p in percentiles]
    })

    st.dataframe(percentileDf)
    st.markdown("Conclusion: The top 1\\% of property prices can be clipped")

    st.markdown("### Descriptive Statistics After Outlier Removal")
    st.dataframe(st.session_state.finalDf.describe())
    
    