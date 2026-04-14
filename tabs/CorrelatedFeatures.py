import streamlit as st
import pandas as pd
import plotly.express as px

def render(df: pd.DataFrame):
    st.markdown("### Correlation Analysis to Find Redundant Features")
    reducedDfNumeric = df.select_dtypes(include='number')
    reducedFeaturesCorr = reducedDfNumeric.corr()
    reducedFeaturesCorrPlot = px.imshow(reducedFeaturesCorr, 
                                        text_auto=".2f", 
                                        color_continuous_scale="RdBu_r", 
                                        aspect="auto")
    reducedFeaturesCorrPlot.update_layout(height=800)
    st.plotly_chart(reducedFeaturesCorrPlot)

    st.markdown("Upon closer examination, it seems like review types are highly correlated with each other, and features indicative of " \
    "property size such as **bedrooms**, **accomodates**, etc. are as well. This is not surprising, however I am opting to keep all " \
    "these features since their correlation coefficients are not extremely high and they might have differing effects on price prediction.")

    st.markdown("### Preview of Final Dataset")
    st.caption(f"{len(df.columns)} features, {len(df)} entries")
    st.dataframe(df.head())