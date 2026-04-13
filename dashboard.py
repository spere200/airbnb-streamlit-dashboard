import streamlit as st

from utils import loadData

# import pages
from sections import DataCleaningSection

st.set_page_config(layout="wide")

# Stylings for subheaders and tabs
with open("styles.css", "r") as f: cssString = f.read()
st.markdown(f"<style>{cssString}</style>", unsafe_allow_html=True)

df = loadData('./data/listings.csv') # initial dataframe

with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.8rem 0;'>
    <div style='font-family:"Playfair Display",serif; font-size:1.25rem;
    color:#f0c040; font-weight:700; line-height:1.4;'>
    Intro to NLP<br>
    </div>
    <div class='tag'><br>FIU - Data Science <br> Prof. Gregory Murad Reis</div>
    </div>
    <hr>

    """, unsafe_allow_html=True)

    page = st.radio("", [
    "Data Cleaning",
    "EDA",
    "Hypothesis Test",
    "Supervised Learning",
    "Unsupervised Learning",
    ], label_visibility="collapsed")

if page == "Data Cleaning":
    DataCleaningSection.render(df)
