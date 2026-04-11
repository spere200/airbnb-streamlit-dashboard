import streamlit as st
import pandas as pd

def render(df: pd.DataFrame):
    st.subheader("Removing Outliers Using the IQR Method")

    st.markdown("To prevent aggressive shrining of the dataset, only select columns will have outliers removed. At the moment, " \
    "I am only targetting columns that are either extremely important to the analysis I want to perform, or columns that showed " \
    "signs of having extreme outliers in the descriptive statistics (std > 5 * mean). The targetted columns are the: ")

    colsToRemoveFrom = [col for col in df.select_dtypes(include='number').columns if df[col].std() > 5 * df[col].mean()]

    with st.container(border=True):
        st.write(colsToRemoveFrom)

    for col in colsToRemoveFrom:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        conditions = (df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)
        removed = (~conditions).sum()
        st.write(f"{col}: {removed} outliers removed.")
        df = df[conditions]

    st.subheader("Preview of Finalized Cleaned Dataframe")
    st.caption(f"{len(df.columns)} features, {len(df)} entries")
    st.dataframe(df.head())

    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe())

    return df