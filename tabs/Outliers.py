import streamlit as st
import pandas as pd

def render(df: pd.DataFrame):
    st.markdown("### Removing Outliers Using the IQR Method")

    st.markdown("To prevent aggressive removal, outliers will only be removed from the price column, since columns such as " \
    "**host_response_rate** had close to 2,000 outliers.")

    colsToRemoveFrom = ['price']
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