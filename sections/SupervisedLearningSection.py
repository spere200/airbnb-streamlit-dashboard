import streamlit as st

from tabs import SupervisedLearning

def render():
    finalDf = st.session_state.finalDf
    finalDf = finalDf.drop(columns=['latitude', 'longitude'])
    SupervisedLearning.render(finalDf)