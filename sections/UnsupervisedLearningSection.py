import streamlit as st

from tabs import UnsupervisedLearning

def render():
    cleanDf = st.session_state.cleanDf
    cleanDf = cleanDf.drop(columns=['latitude', 'longitude'])
    UnsupervisedLearning.render(cleanDf)