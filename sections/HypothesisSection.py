import streamlit as st

from tabs import HypothesisTesting

def render():
    finalDf = st.session_state.finalDf
    HypothesisTesting.render(finalDf)