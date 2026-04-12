import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.preprocessing import MinMaxScaler

def render(df: pd.DataFrame):
    dataPreprocessingTab, modelPredictionTab = st.tabs(["Data Preprocessing", "Model Training and Predictions"])

    with dataPreprocessingTab:
        st.subheader("Data Preprocessing")
        st.markdown("#### Preview of Current Dataframe")
        st.caption(f"{len(df.columns)} features, {len(df)} entries")
        st.write(df.head())

        st.markdown("#### Creating an Encoded Dataframe for Linear Regression and KNN")
        st.markdown("**Steps**: Nomalize numeric columns, One-Hot Encode categorical " \
        "columns, and convert boolean columns to 0/1.") 

        encodedDf = df.copy()

        # Normalize numeric columns
        numericCols = encodedDf.select_dtypes(include='number').columns
        scaler = MinMaxScaler()
        encodedDf[numericCols] = scaler.fit_transform(encodedDf[numericCols])

        # One-Hot encode categorical columns
        categoricalCols = encodedDf.select_dtypes(include='object').columns
        encodedDf = pd.get_dummies(encodedDf, columns=categoricalCols)

        # Convert boolean columns to 0/1
        booleanCols = encodedDf.select_dtypes(include='bool').columns
        encodedDf[booleanCols] = encodedDf[booleanCols].astype(int)

        st.caption(f"{len(encodedDf.columns)} features, {len(encodedDf)} entries")
        st.dataframe(encodedDf.head())


        

