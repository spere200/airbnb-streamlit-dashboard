import streamlit as st

from utils import loadData

# import sections
from sections import DataCleaningSection
from sections import EDASection
from sections import HypothesisSection
from sections import SupervisedLearningSection
from sections import UnsupervisedLearningSection

st.set_page_config(layout="wide")

import plotly.io as pio
pio.templates.default = "plotly_dark"

# Stylings for subheaders and tabs
with open("styles.css", "r") as f: cssString = f.read()
st.markdown(f"<style>{cssString}</style>", unsafe_allow_html=True)

df = loadData('./data/listings.csv') # initial dataframe

with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.8rem 0;'>
        <div style='font-family:"Playfair Display",serif; font-size:1.5rem;
            color:#f0c040; font-weight:700; line-height:1.5;'>
        Broward County Airbnb Listings Dashboard<br>
        </div>
    </div>
    <hr>

    """, unsafe_allow_html=True)

    page = st.radio("", [
    "Data Cleaning",
    "EDA",
    "Hypothesis Test",
    "Supervised Learning",
    "Unsupervised Learning",
    ], index=0, label_visibility="collapsed") 

if page == "Data Cleaning":
    DataCleaningSection.render(df)

elif page == "EDA":
    EDASection.render()

elif page == "Hypothesis Test":
    HypothesisSection.render()

elif page == "Supervised Learning":
    SupervisedLearningSection.render()

elif page == "Unsupervised Learning":
    UnsupervisedLearningSection.render()
