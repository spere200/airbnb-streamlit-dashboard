import streamlit as st
import pandas as pd

def render(df: pd.DataFrame):
    st.markdown("### Removing Outliers Using the IQR Method")

    st.markdown("I had to be careful with outlier removal since some bounded features such as **host_acceptance_rate** " \
    "have thousands of outliers, and other features like **bedrooms**, despite being unbounded, should be kept, since " \
    "I don't want to overfit to the \"most average\" listing for the sake of model performance. As such, I only removed " \
    "outliers from **price** and **minimum_nights**. Both features had extremely outliers that were heavily skewing the " \
    "distribution.")

    # handling outliers was a pretty involved process. using a simple IQR approach would remove legitimate expensive properties,
    # which would cause the training models to overfit to a specific subset that does not fairly represent the dataset.

    colsToRemoveFrom = ['price'] 
    colsToRemoveFrom = [] 
    for col in colsToRemoveFrom:
    # for col in df.select_dtypes(include='number').columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        conditions = (df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)
        removed = (~conditions).sum()
        # if removed > 0 and removed <= 200 or col == 'price':

        st.markdown(f"* **{col}**: {removed} outliers removed.")
        df = df[conditions]

    st.markdown("### Descriptive Statistics for Dataset With Outliers Removed")
    st.caption(f"{len(df.columns)} features, {len(df)} entries")
    st.dataframe(df.describe())

    return df