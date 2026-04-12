import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.cluster import KMeans

def render(df: pd.DataFrame):
    st.subheader("Using K-Means Clustering to Group Neighborhoods by Average Price")
    

    

