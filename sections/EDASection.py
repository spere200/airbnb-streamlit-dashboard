import streamlit as st

from tabs import Charts

def render():
    cleanDf = st.session_state.cleanDf
    Charts.render(cleanDf)
    