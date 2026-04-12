import streamlit as st
import pandas as pd
import plotly.express as px

import numpy as np

# import math

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import MinMaxScaler

def render(df: pd.DataFrame):
    dataPreprocessingTab, modelPredictionTab = st.tabs(['Data Preprocessing', 'Model Training and Predictions'])

    with dataPreprocessingTab:
        st.subheader('Data Preprocessing')
        st.markdown('#### Preview of Current Dataframe')
        st.caption(f'{len(df.columns)} features, {len(df)} entries')
        st.write(df.head())

        st.markdown('#### Creating an Encoded Dataframe for Linear Regression and KNN')
        st.markdown('**Steps**: Nomalize numeric columns, One-Hot Encode categorical ' \
        'columns, and convert boolean columns to 0/1.') 

        encodedDf = df.copy()

        # decided to keep outliers and log transform price, this significantly improved performance
        encodedDf['price'] = np.log1p(encodedDf['price'])

        # Normalize numeric columns
        numericCols = encodedDf.select_dtypes(include='number').columns
        scaler = MinMaxScaler()
        encodedDf[numericCols] = scaler.fit_transform(encodedDf[numericCols])

        # One-Hot encode categorical columns
        categoricalCols = encodedDf.select_dtypes(include='object').columns
        encodedDf = pd.get_dummies(encodedDf, columns=categoricalCols)

        # Convert boolean columns to 0/1
        booleanCols = encodedDf.select_dtypes(include='bool').columns
        encodedDf[booleanCols] = encodedDf[booleanCols].astype(int)

        st.caption(f'{len(encodedDf.columns)} features, {len(encodedDf)} entries')
        st.dataframe(encodedDf.head())

    with modelPredictionTab:
        st.subheader('Model Predictions and Performance Assessment')

        # create a price scaler to report actual results instead of scaled results
        priceScaler = MinMaxScaler()
        priceScaler.fit(df[['price']])

        # creating the train test split
        X = encodedDf.drop(columns=['price'])
        y = encodedDf['price']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_test_scaled = priceScaler.inverse_transform(y_test.values.reshape(-1, 1)).flatten()

        with st.container(border=True):
            lrGraphCol, _, lrEvalCol= st.columns([10, 1, 10])

            with lrGraphCol:
                linRegModel = LinearRegression()
                linRegModel.fit(X_train, y_train)

                y_pred_linreg = linRegModel.predict(X_test)
                y_pred_linreg = y_pred_linreg.clip(0, y_pred_linreg.max()) # clip predictions so negative dollar amounts aren't shown
                y_pred_linreg_scaled = priceScaler.inverse_transform(y_pred_linreg.reshape(-1, 1)).flatten()
                
                sortedLinRegResults = pd.DataFrame({'Actual Values': y_test_scaled, 
                                            'Predicted Values': y_pred_linreg_scaled}
                                            ).sort_values('Actual Values').reset_index(drop=True)
                linRegPlot = px.line(sortedLinRegResults, y=["Predicted Values", "Actual Values"], 
                                        color_discrete_map={'Actual Values': 'green', 'Predicted Values': 'lightblue'})
                linRegPlot.update_layout(title="Linear Regression")
                st.plotly_chart(linRegPlot)

            with lrEvalCol:
                st.markdown("##### Model Evaluation")

            # featureImportance = pd.DataFrame({'Feature': X.columns, 
            #                                     'Coefficient': linRegModel.coef_}
            #                                     ).sort_values('Coefficient', ascending=False)
            # st.dataframe(featureImportance)

        # # Cross validation to check best value for n, result was any value from 4 to 7, choosing 4
        # for n in range(1, 101):
        #     knn = KNeighborsRegressor(n_neighbors=k)
        #     scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='r2')
        #     print(f"k={k}: {scores.mean():.2f}")

        with st.container(border=True):
            knnGraphCol, _, knnEvalCol= st.columns([10, 1, 10])

            with knnGraphCol:
                knnModel = KNeighborsRegressor(n_neighbors=4)
                knnModel.fit(X_train, y_train)

                y_pred_knn = knnModel.predict(X_test)
                y_pred_knn.clip(0, y_pred_knn.max())
                y_pred_knn_scaled = priceScaler.inverse_transform(y_pred_knn.reshape(-1, 1)).flatten()

                sortedKnnResults = pd.DataFrame({'Actual Values': y_test_scaled, 
                                            'Predicted Values': y_pred_knn_scaled}
                                            ).sort_values('Actual Values').reset_index(drop=True)
                
                knnPlot = px.line(sortedKnnResults, y=["Predicted Values", "Actual Values"], 
                                        color_discrete_map={'Actual Values': 'green', 'Predicted Values': 'lightblue'})
                knnPlot.update_layout(title="KNN Regrerssion")
                st.plotly_chart(knnPlot)

            with knnEvalCol:
                st.markdown("##### Model Evaluation")

        with st.container(border=True):
            svrGraphCol, _, svrEvalCol= st.columns([10, 1, 10])

            with svrGraphCol:
                svrModel = SVR()
                svrModel.fit(X_train, y_train)

                y_pred_svr = svrModel.predict(X_test)
                y_pred_svr.clip(0, y_pred_svr.max())
                y_pred_svr_scaled = priceScaler.inverse_transform(y_pred_svr.reshape(-1, 1)).flatten()

                sortedSvrResults = pd.DataFrame({'Actual Values': y_test_scaled, 
                                            'Predicted Values': y_pred_svr_scaled}
                                            ).sort_values('Actual Values').reset_index(drop=True)
                
                svrPlot = px.line(sortedSvrResults, y=["Predicted Values", "Actual Values"], 
                                        color_discrete_map={'Actual Values': 'green', 'Predicted Values': 'lightblue'})
                svrPlot.update_layout(title="Support Vector Regrerssion")
                st.plotly_chart(svrPlot)

            with svrEvalCol:
                st.markdown("##### Model Evaluation")

        with st.container(border=True):
            dtGraphCol, _, dtEvalCol= st.columns([10, 1, 10])

            with dtGraphCol:
                treeModel = DecisionTreeRegressor(random_state=42)
                treeModel.fit(X_train, y_train)
                y_pred_tree = treeModel.predict(X_test)
                y_pred_tree.clip(0, y_pred_tree.max())
                y_pred_tree_scaled = priceScaler.inverse_transform(y_pred_tree.reshape(-1, 1)).flatten()

                sortedTreeResults = pd.DataFrame({'Actual Values': y_test_scaled, 
                                            'Predicted Values': y_pred_tree_scaled}
                                            ).sort_values('Actual Values').reset_index(drop=True)
                
                treePlot = px.line(sortedTreeResults, y=["Predicted Values", "Actual Values"], 
                                        color_discrete_map={'Actual Values': 'green', 'Predicted Values': 'lightblue'})
                treePlot.update_layout(title="Decision Tree Regrerssion")
                st.plotly_chart(treePlot)

            with dtEvalCol:
                st.markdown("##### Model Evaluation")

        with st.container(border=True):
            rfGraphCol, _, rfEvalCol= st.columns([10, 1, 10])

            with rfGraphCol:
                forestModel = RandomForestRegressor(n_estimators=100, random_state=42)
                forestModel.fit(X_train, y_train)
                y_pred_forest = forestModel.predict(X_test)
                y_pred_forest.clip(0, y_pred_forest.max())
                y_pred_forest_scaled = priceScaler.inverse_transform(y_pred_forest.reshape(-1, 1)).flatten()

                sortedforestResults = pd.DataFrame({'Actual Values': y_test_scaled, 
                                            'Predicted Values': y_pred_forest_scaled}
                                            ).sort_values('Actual Values').reset_index(drop=True)
                
                forestPlot = px.line(sortedforestResults, y=["Predicted Values", "Actual Values"], 
                                        color_discrete_map={'Actual Values': 'green', 'Predicted Values': 'lightblue'})
                forestPlot.update_layout(title="Random Forest Regrerssion")
                st.plotly_chart(forestPlot)

            with rfEvalCol:
                st.markdown("##### Model Evaluation")