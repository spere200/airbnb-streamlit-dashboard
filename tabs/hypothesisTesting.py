import streamlit as st
import pandas as pd
import math
from scipy.stats import norm
import numpy as np

def render(df: pd.DataFrame):
    st.markdown("### Hypothesis")
    st.markdown("**H<sub>0</sub>**: There is no significant difference in mean price between superhost and regular host listings.", unsafe_allow_html=True)
    st.markdown("**H<sub>a</sub>**: Superhost listings have a higher average price than listings by regular hosts.", unsafe_allow_html=True)
    
    superhostPrices = df[df['host_is_superhost'] == True]['price']
    regularPrices = df[df['host_is_superhost'] == False]['price']

    st.markdown("### Two-Sample, One-Tailed Z-Test")
    st.write("Since the sample size is large enough (8,506 entries), I am performing a Z-test.")
    nSuperhost, meanSuperhost, stdSuperhost = len(superhostPrices), superhostPrices.mean(), superhostPrices.std()
    nRegular, meanRegular, stdRegular = len(regularPrices), regularPrices.mean(), regularPrices.std()

    _, col1, col2, col3, _ = st.columns([2, 1, 1, 1, 2])

    with col1:
        st.markdown("##### Superhost Stats")
        st.markdown(f"**n<sub>1<sub>** = {nSuperhost}", unsafe_allow_html=True)
        st.markdown(f"**x̄<sub>1<sub>** = {meanSuperhost:.4f}", unsafe_allow_html=True)
        st.markdown(f"**s<sub>1<sub>** = {stdSuperhost:.4f}", unsafe_allow_html=True)

    with col2:
        st.markdown("##### Regular Host Stats")
        st.markdown(f"**n<sub>2<sub>** = {nRegular}", unsafe_allow_html=True)
        st.markdown(f"**x̄<sub>2<sub>** = {meanRegular:.4f}", unsafe_allow_html=True)
        st.markdown(f"**s<sub>2<sub>** = {stdRegular:.4f}", unsafe_allow_html=True)

    standardError = np.sqrt(stdSuperhost**2/nSuperhost + stdRegular**2/nRegular)
    zScore = (meanSuperhost - meanRegular)/standardError
    pValue = 1 - norm.cdf(zScore)

    with col3:
        st.markdown("##### Test Results")
        st.markdown(f"**Standard Error** = {standardError:.2f}", unsafe_allow_html=True)
        st.markdown(f"**Z-Score** = {zScore:.4f}", unsafe_allow_html=True)
        st.markdown(f"**P-Value** = {pValue:.4f}", unsafe_allow_html=True)

    st.markdown("<p style='text-align: center'>With a <b>p-value</b> of approximately 0, there is extremely " \
    "strong evidence to reject the null hypothesis. We can safely conclude that superhost listings have a " \
    "higher average price than regular host listings.</p>", unsafe_allow_html=True)

    