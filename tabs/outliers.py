import streamlit as st
import pandas as pd

def render(df: pd.DataFrame):
    st.markdown("### Removing Outliers Using the IQR Method")

    st.markdown("After testing regression models, I found much better results by keeping outliers in and performing " \
    "a log transformation on numeric columns. As such, there was no need to remove outliers for the training set. " \
    "However, for the charts tab, to better show the distribution of data, I am removing all outliers from the dataset " \
    "using the IQR method. To prevent aggressive removal, outliers will only be removed from a column below a certain "
    "threshold (currently 400) or any outliers in price, since price is the feature of interest for my hypothesis test. " \
    "The following outliers were removed:")

    # colsToRemoveFrom = [col for col in df.select_dtypes(include='number').columns if df[col].std() > 5 * df[col].mean()]

    # with st.container(border=True):
    #     st.write(colsToRemoveFrom)

    # for col in colsToRemoveFrom:
    for col in df.select_dtypes(include='number').columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        conditions = (df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)
        removed = (~conditions).sum()
        if removed > 0 and removed <= 400 or col == 'price':
            st.markdown(f"* **{col}**: {removed} outliers removed.")
            df = df[conditions]

    st.markdown("### Descriptive Statistics for Dataset With Outliers Removed")
    st.caption(f"{len(df.columns)} features, {len(df)} entries")
    st.dataframe(df.describe())

    return df