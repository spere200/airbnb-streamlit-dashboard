import streamlit as st
import pandas as pd
import plotly.express as px

def render(df: pd.DataFrame):
    st.markdown("### List of Non-Numeric Features")
    nonNumericColumns = df.select_dtypes(exclude=["number"]).dtypes
    st.dataframe(nonNumericColumns)

    st.markdown("### Converting string Representation of Numbers to int/float")
    st.markdown('**host_response_rate**, **host_acceptance_rate**, and **price** are currently represented ' \
    'as strings. Converting to int/float:')

    df['host_acceptance_rate'] = df['host_acceptance_rate'].str.rstrip('%').astype(float) / 100
    df['host_response_rate'] = df['host_response_rate'].str.rstrip('%').astype(float) / 100
    df['price'] = df['price'].str.replace('[$,]', '', regex=True).astype(float)

    st.dataframe(df.head())

    st.markdown("### Handling Categorical Columns")
    st.markdown("#### Identifying Category Type")
    st.write("")

    # recalculating since values above were recently converted to numbers
    nonNumericColumns = df.select_dtypes(exclude=["number"]).dtypes
    nonBinaryColumns = {}
    binaryColumns = {}

    for col in nonNumericColumns.index:
        if df[col].nunique() > 2:
            nonBinaryColumns[col] = df[col].nunique()
        else:
            binaryColumns[col] = df[col].nunique()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Binary Categorical Columns (+ Unique Values)")
        with st.container(border=True, height=220):
            st.write(binaryColumns)

    with col2:
        st.markdown("##### Non-Binary Categorical Columns (+ Unique Values)")
        with st.container(border=True, height=220):
            st.write(nonBinaryColumns)

    del nonBinaryColumns['amenities'] # too many unique values, need to find a better way of handling this category

    st.markdown("#### Unique Values of Non-Binary Categorical Columns")
    st.caption('Skipping "amenities" since it has too many unique values.')
    for col, container in zip(nonBinaryColumns.keys(), st.columns(4)):
        if col == "amenities":
            continue

        with container:
            st.markdown(f"##### {col}")
            st.write(df[col].unique())

    st.markdown("#### Converting Categorical Column")
    st.markdown("* **t**/**f** binary columns can all be converted to boolean columns.")
    st.markdown("* **neighbourhood_cleansed** can remain as is.")
    st.markdown("* **property_type** is too inconsistent, with some values even being in different languages. Dropping the column.")
    st.markdown("* **room_type** can remain as is.")
    st.markdown("* **bathroom_text** can have its number removed and can be converted to **private_bathroom** with values "
    "**t** or **f**, since **bathrooms** already contains the number of bathrooms.")
    st.markdown("* **amenities** has far too many unique values as it stands. I'll perform a correlation analysis to find the most " \
    "relevant amenities and separate them all into their own column. ")

    # converting binary columns to boolean
    for col in binaryColumns.keys():
        df[col] = df[col].str.contains('t', case=False, na=False)

    # dropping property type
    df = df.drop(columns=["property_type"])

    # converting bathroom_text to private_bathroom
    df = df.rename(columns={'bathrooms_text': 'private_bathroom'})
    df['private_bathroom'] = ~df['private_bathroom'].str.contains('shared', case=False, na=False)

    st.markdown("#### Dealing With Amenities")
    st.markdown("Upon closer examination, the values of amenities are far too granular, sometimes it even lists the brand of the TV " \
    "found in the property. As such, I can't process the entire list. The two approaches I can think of at the time are to either " \
    "remove amenities altogether, or convert the list of amenities to the number of amenities. Neither would be ideal, since number " \
    "of amenities would reward listings that exaggerate and list even the most minimal things as amenities, but the alternative would " \
    "be to lose data that could be significant. However, since hosts that try to exaggerate their amenities might be trying to make " \
    "up for a low quality listing, it could be that a higher number of amenities is seen as an indicator of a lower quality property.")
    st.markdown("As such I am going to convert amenities to number of amenities, and perform a correlation analysis to see how number " \
    "of amenities correlates with prices, review scores, bookings, etc..")

    # converting amenities list to number of amenities
    df['amenities'] = df['amenities'].apply(eval).apply(len)

    st.markdown("##### Correlation Heatmap for Amenities")
    correlationColumns = ['amenities', 'price', 'estimated_occupancy_l365d', 'review_scores_rating', 
                          'review_scores_accuracy', 'review_scores_value']
    corrMatrix = df[correlationColumns].corr().iloc[[0]]
    corrHeatMap = px.imshow(corrMatrix, 
                    text_auto=".2f",
                    color_continuous_scale="RdBu_r")
    st.plotly_chart(corrHeatMap)
    
    st.write("It looks like the number of amenities is not correlated with any relevant value. As such, it should be safe to drop amenities.")
    
    finalDf = df.drop(columns=["amenities"])

    st.markdown("### Preview of Dataframe With Finalized Feature Set")
    st.caption(f"{len(finalDf.columns)} features, {len(finalDf)} entries")
    st.dataframe(finalDf.head())

    st.markdown("### Descriptive Statistics")
    st.dataframe(finalDf.describe())

    return finalDf
