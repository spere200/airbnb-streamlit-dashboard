import streamlit as st

from tabs import HypothesisTesting

def render():
    cleanDf = st.session_state.cleanDf
    HypothesisTesting.render(cleanDf)