import streamlit as st

from tabs import UnsupervisedLearning

def render():
    finalDf = st.session_state.finalDf
    finalDf = finalDf.drop(columns=['latitude', 'longitude'])
    UnsupervisedLearning.render(finalDf)