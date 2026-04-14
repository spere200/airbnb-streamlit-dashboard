import streamlit as st

from tabs import SupervisedLearning

def render():
    cleanDf = st.session_state.cleanDf
    cleanDf = cleanDf.drop(columns=['latitude', 'longitude'])
    SupervisedLearning.render(cleanDf)