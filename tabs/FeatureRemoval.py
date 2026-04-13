import streamlit as st
import pandas as pd
import plotly.express as px

def render(df: pd.DataFrame):
    st.markdown("### Preview of Dataframe After Handling Missing Values")
    st.caption(f"{len(df.columns)} features, {len(df)} entries")
    st.dataframe(df.head())

    st.markdown("### Removing Unexplained/Irrelevant Features")
    col1, col2, col3 = st.columns(3)

    with col1:
        irrelevantCols = ['host_verifications', 'number_of_reviews_ltm', 'number_of_reviews_l30d', 'number_of_reviews_ly',
                          'reviews_per_month', 'availability_30', 'availability_60', 'availability_90', 'availability_eoy', 
                          'host_listings_count', 'host_has_profile_pic', 'number_of_reviews_ly']
        
        st.write("##### Removing Irrelevant Features")
        with st.container(border=True, height=400):    
            st.markdown('* **host_verifications** contains no empty rows since verification appears to be mandatory, and ' \
            '**host_identity_verified** contains much more important information than just verification method.')
            st.markdown("* Since this analysis focuses on general information about the properties, granular data such as" \
            " **number_of_reviews_l30d** can be removed when more general information such as **number_of_reviews** exists.")
            st.markdown("* Some columns (such as **host_has_profile_pic**) are of no interest.")

            st.write("Affected columns:")
            st.write(irrelevantCols)

        df = df.drop(columns=irrelevantCols, axis=1)

    with col2:
        unexplainedCols = ["minimum_minimum_nights", "maximum_minimum_nights", "minimum_maximum_nights", "maximum_maximum_nights",
                           "minimum_nights_avg_ntm", "maximum_nights_avg_ntm", "calculated_host_listings_count_entire_homes",
                           "calculated_host_listings_count_private_rooms", "calculated_host_listings_count_shared_rooms", 
                           "calculated_host_listings_count"]

        st.write("##### Removing Unexplained Features")
        with st.container(border=True, height=400):    
            st.write("The following features are not explained anywhere in the dataset source, and their meaning cannot " \
            "be extrapolated from their name. As such, I am opting for their removal:")
            st.write(unexplainedCols)

        df = df.drop(columns=unexplainedCols, axis=1)

    with col3:
        noVarianceCols = [col for col in df.columns if df[col].nunique() == 1]

        st.write()

        st.write("##### Removing Columns With No Variance")
        with st.container(border=True, height=400):    
            st.write("The following columns only have a single value across all rows:")
            st.write(noVarianceCols)

        dfReducedFeatures = df.drop(columns=noVarianceCols, axis=1)

    st.markdown("### Correlation Analysis to Find Redundant Features")
    reducedDfNumeric = dfReducedFeatures.select_dtypes(include='number')
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

    st.markdown("### Preview of Final Dataset With All Features of Interest")
    st.caption(f"{len(dfReducedFeatures.columns)} features, {len(dfReducedFeatures)} entries")
    st.dataframe(dfReducedFeatures.head())

    return dfReducedFeatures