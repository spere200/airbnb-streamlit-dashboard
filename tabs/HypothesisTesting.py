import streamlit as st
import pandas as pd
from scipy.stats import norm
import numpy as np

def render(df: pd.DataFrame):
    st.markdown('### Hypothesis')
    st.markdown('**H<sub>0</sub>**: There is no significant difference in mean price between listings with higher '
    'and lower estimated occupancy.', 
                unsafe_allow_html=True)
    st.markdown('**H<sub>a</sub>**: Listings with higher estimated occupancy have a lower average price.', unsafe_allow_html=True)

    medianOccupancy = df['estimated_occupancy_l365d'].median()
    highOccupancy = df[df['estimated_occupancy_l365d'] > medianOccupancy]['price']
    lowOccupancy = df[df['estimated_occupancy_l365d'] <= medianOccupancy]['price']

    st.markdown('### Two-Sample, One-Tailed Z-Test')
    st.write(f"""Listings are split by median estimated occupancy ({medianOccupancy:.0f} nights/year).
             Since the sample size is large enough ({len(df):,} entries), I am performing a Z-test.""")

    nHigh, meanHigh, stdHigh = len(highOccupancy), highOccupancy.mean(), highOccupancy.std()
    nLow, meanLow, stdLow = len(lowOccupancy), lowOccupancy.mean(), lowOccupancy.std()

    _, col1, col2, col3, _ = st.columns([2, 1, 1, 1, 2])
    with col1:
        st.markdown('##### High Occupancy Stats')
        st.markdown(f'**n<sub>1</sub>** = {nHigh}', unsafe_allow_html=True)
        st.markdown(f'**x̄<sub>1</sub>** = ${meanHigh:.2f}', unsafe_allow_html=True)
        st.markdown(f'**s<sub>1</sub>** = ${stdHigh:.2f}', unsafe_allow_html=True)
    with col2:
        st.markdown('##### Low Occupancy Stats')
        st.markdown(f'**n<sub>2</sub>** = {nLow}', unsafe_allow_html=True)
        st.markdown(f'**x̄<sub>2</sub>** = ${meanLow:.2f}', unsafe_allow_html=True)
        st.markdown(f'**s<sub>2</sub>** = ${stdLow:.2f}', unsafe_allow_html=True)

    standardError = np.sqrt(stdHigh**2/nHigh + stdLow**2/nLow)
    # One-tailed: testing if high occupancy mean is LOWER than low occupancy mean
    zScore = (meanHigh - meanLow) / standardError
    pValue = norm.cdf(zScore)

    with col3:
        st.markdown('##### Test Results')
        st.markdown(f'**Standard Error** = {standardError:.4f}', unsafe_allow_html=True)
        st.markdown(f'**Z-Score** = {zScore:.4f}', unsafe_allow_html=True)
        st.markdown(f'**P-Value** = {pValue:.10f}', unsafe_allow_html=True)

    if pValue < 0.05:
        conclusion = (f'With a <b>p-value</b> of {pValue:.10f}, there is strong evidence to reject the null hypothesis. '
                      'We can conclude that listings with higher estimated occupancy have a lower average price '
                      f'(${meanHigh:.2f}) compared to listings with lower occupancy (${meanLow:.2f}).')
    else:
        conclusion = (f'With a <b>p-value</b> of {pValue:.10f}, there is insufficient evidence to reject the null hypothesis. '
                      f'We cannot conclude that higher occupancy listings have a lower average price.')

    st.markdown(f'<p style="text-align: center">{conclusion}</p>', unsafe_allow_html=True)