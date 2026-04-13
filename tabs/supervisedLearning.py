import streamlit as st
import pandas as pd
import plotly.express as px

import numpy as np

import pickle
import os
# import math

from sklearn.model_selection import train_test_split#, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def getModel(name, model, X_train, y_train):
    path = os.path.join('models', f"{name}.pkl")

    # if model exists, fetch it, otherwise train it and save it
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    model.fit(X_train, y_train)
    os.makedirs('models', exist_ok=True)

    with open(path, 'wb') as f:
        pickle.dump(model, f)

    return model


def render(df: pd.DataFrame):
    (dataPreprocessingTab, 
     modelPredictionTab, 
     significantFeaturesTab) = st.tabs(['Data Preprocessing', 
                                        'Model Training and Predictions',
                                        'Most Significant Features'])

    with dataPreprocessingTab:
        st.subheader('Data Preprocessing')
        st.markdown('#### Preview of Current Dataframe')
        st.caption(f'{len(df.columns)} features, {len(df)} entries')
        st.write(df.head())

        st.markdown('#### Creating a One-Hot Encoded, Log Transformed, and Normalized Dataframe for Linear ' \
        'Regression, KNN, and SVR')
        st.markdown('**Steps**: Nomalize numeric columns, One-Hot Encode categorical ' \
        'columns, and convert boolean columns to 0/1.') 

        encodedDf = df.copy()

        # decided to keep outliers and log transform price, this significantly improved performance
        encodedDf['price'] = np.log1p(encodedDf['price'])
        treeDf = encodedDf.copy() # dataframe for trees, copying after log transform but before normalization

        # Normalize numeric columns
        # create a price scaler to report actual results instead of scaled results
        # this scaler is trained on the log transformed price values
        priceScaler = MinMaxScaler()
        priceScaler.fit(encodedDf[['price']])

        # create a minmax scaler for all numeric cols and scale them
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

        st.markdown('#### Creating a Label Encoded Dataframe with Log Transformed Price for Random Forest and XGBoost') 
        # treeDf  = pd.get_dummies(treeDf, columns=categoricalCols)
        # st.dataframe(treeDf.head())

        # Label Encode categorical variables
        for col in treeDf.select_dtypes(include='object').columns:
            treeDf[col] = LabelEncoder().fit_transform(treeDf[col].astype(str))

        st.caption(f'{len(treeDf.columns)} features, {len(treeDf)} entries')
        st.dataframe(treeDf.head())

    with modelPredictionTab:
        st.subheader('Model Predictions and Performance Assessment') 

        # creating the train test split for linear regression, knn, and svr
        X = encodedDf.drop(columns=['price'])
        y = encodedDf['price']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        # getting the actual dollar amount of the test set for nore explainable metrics
        yTestUnscaled = priceScaler.inverse_transform(y_test.values.reshape(-1, 1)).flatten() # undo normalization
        yTestDollars = np.expm1(yTestUnscaled) # undo log transform

        with st.container(border=True):
            lrGraphCol, _, lrEvalCol= st.columns([10, 1, 6])

            with lrGraphCol:
                linRegModel = getModel('linear_regression.pkl', LinearRegression(), 
                                       X_train=X_train, y_train=y_train)

                y_pred_linreg = linRegModel.predict(X_test)
                y_pred_linreg = y_pred_linreg
                lrPredDollars = priceScaler.inverse_transform(y_pred_linreg.reshape(-1, 1)).flatten()# undo normalization
                lrPredDollars = np.expm1(lrPredDollars) # undo log transform
                
                sortedLinRegResults = pd.DataFrame({'Actual Values': yTestDollars,
                                            'Predicted Values': lrPredDollars}
                                            ).sort_values('Actual Values').reset_index(drop=True)
                linRegPlot = px.line(sortedLinRegResults, y=["Predicted Values", "Actual Values"], 
                                        color_discrete_map={'Actual Values': 'green', 'Predicted Values': 'lightblue'})
                linRegPlot.update_layout(title="Linear Regression")
                st.plotly_chart(linRegPlot)

            with lrEvalCol:
                lregR2 = r2_score(y_test, y_pred_linreg)
                lregRMSE = np.sqrt(mean_squared_error(yTestDollars, lrPredDollars))

                st.markdown("##### Model Evaluation")
                st.markdown(f"**R<sup>2</sup>** = {lregR2:.4f}", unsafe_allow_html=True)
                st.markdown(f"**RMSE** = ${lregRMSE:.2f}", unsafe_allow_html=True)

                st.markdown("###### While the R<sup>2</sup> of this model is decent, its RMSE " \
                "is far too high. This shows that while the model captured the general trend of the " \
                "data, its average guesses are far from the mean, and it performs extremely poorly for the "
                "more expensive properties as can be observed in the graph.", 
                unsafe_allow_html=True)

        # # Cross validation to check best value for n, result was any value from 4 to 7, choosing 4
        # for n in range(1, 101):
        #     knn = KNeighborsRegressor(n_neighbors=k)
        #     scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='r2')
        #     print(f"k={k}: {scores.mean():.2f}")

        with st.container(border=True):
            knnGraphCol, _, knnEvalCol= st.columns([10, 1, 6])

            with knnGraphCol:
                knnModel = getModel('knn_regression.pkl', KNeighborsRegressor(n_neighbors=4), 
                                       X_train=X_train, y_train=y_train)

                y_pred_knn = knnModel.predict(X_test)
                y_pred_knn.clip(0, y_pred_knn.max())
                knnPredDollars = priceScaler.inverse_transform(y_pred_knn.reshape(-1, 1)).flatten()# undo normalization
                knnPredDollars = np.expm1(knnPredDollars) # undo log transform

                sortedKnnResults = pd.DataFrame({'Actual Values': yTestDollars, 
                                            'Predicted Values': knnPredDollars}
                                            ).sort_values('Actual Values').reset_index(drop=True)
                
                knnPlot = px.line(sortedKnnResults, y=["Predicted Values", "Actual Values"], 
                                        color_discrete_map={'Actual Values': 'green', 'Predicted Values': 'lightblue'})
                knnPlot.update_layout(title="KNN Regression")
                st.plotly_chart(knnPlot)

            with knnEvalCol:
                knnR2 = r2_score(y_test, y_pred_knn)
                knnRMSE = np.sqrt(mean_squared_error(yTestDollars, knnPredDollars))

                st.markdown("##### Model Evaluation")
                st.markdown(f"**R<sup>2</sup>** = {knnR2:.4f}", unsafe_allow_html=True)
                st.markdown(f"**RMSE** = ${knnRMSE:.2f}", unsafe_allow_html=True)

                st.markdown("###### Worst performance in both metrics. It seems like the current dataset does not" \
                "work very well for this model.", 
                unsafe_allow_html=True)

        with st.container(border=True):
            svrGraphCol, _, svrEvalCol = st.columns([10, 1, 6])

            with svrGraphCol:
                svrModel = getModel('svr_regression.pkl', SVR(), 
                                       X_train=X_train, y_train=y_train)

                y_pred_svr = svrModel.predict(X_test)
                y_pred_svr.clip(0, y_pred_svr.max())
                svrPredDollars = priceScaler.inverse_transform(y_pred_svr.reshape(-1, 1)).flatten()# undo normalization
                svrPredDollars = np.expm1(svrPredDollars) # undo log transform

                sortedSvrResults = pd.DataFrame({'Actual Values': yTestDollars, 
                                            'Predicted Values': svrPredDollars}
                                            ).sort_values('Actual Values').reset_index(drop=True)
                
                svrPlot = px.line(sortedSvrResults, y=["Predicted Values", "Actual Values"], 
                                        color_discrete_map={'Actual Values': 'green', 'Predicted Values': 'lightblue'})
                svrPlot.update_layout(title="Support Vector Regression")
                st.plotly_chart(svrPlot)

            with svrEvalCol:
                svrR2 = r2_score(y_test, y_pred_svr)
                svrRMSE = np.sqrt(mean_squared_error(yTestDollars, svrPredDollars))

                st.markdown("##### Model Evaluation")
                st.markdown(f"**R<sup>2</sup>** = {svrR2:.4f}", unsafe_allow_html=True)
                st.markdown(f"**RMSE** = ${svrRMSE:.2f}", unsafe_allow_html=True)

                st.markdown("###### Best performance of all the non-tree based models. Its predictions graph looks " \
                "similar to the linear regression graph, but it does a better job at handling expensive properties.", 
                unsafe_allow_html=True)

        # creating the train test split for decision tree, random forest, and xgboost
        X = treeDf.drop(columns=['price']) 
        y = treeDf['price'] 

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        with st.container(border=True):
            dtGraphCol, _, dtEvalCol= st.columns([10, 1, 6])

            with dtGraphCol:
                treeModel = getModel('dt_regression.pkl', DecisionTreeRegressor(random_state=42), 
                                       X_train=X_train, y_train=y_train)
                
                y_pred_tree = treeModel.predict(X_test)
                y_pred_tree.clip(0, y_pred_tree.max())
                treePredDollars = np.expm1(y_pred_tree) # undo log transform

                sortedTreeResults = pd.DataFrame({'Actual Values': yTestDollars, 
                                            'Predicted Values': treePredDollars}
                                            ).sort_values('Actual Values').reset_index(drop=True)
                
                treePlot = px.line(sortedTreeResults, y=["Predicted Values", "Actual Values"], 
                                        color_discrete_map={'Actual Values': 'green', 'Predicted Values': 'lightblue'})
                treePlot.update_layout(title="Decision Tree Regression")
                st.plotly_chart(treePlot)

            with dtEvalCol:
                treeR2 = r2_score(y_test, y_pred_tree)
                treeRMSE = np.sqrt(mean_squared_error(yTestDollars, treePredDollars))

                st.markdown("##### Model Evaluation")
                st.markdown(f"**R<sup>2</sup>** = {treeR2:.4f}", unsafe_allow_html=True)
                st.markdown(f"**RMSE** = ${treeRMSE:.2f}", unsafe_allow_html=True)

                st.markdown("###### This single-tree based model shows the highest R<sup>2</sup> yet, " \
                "with the best RMSE so far. However, the graph does seem to have a lot of noise.",
                unsafe_allow_html=True)

        with st.container(border=True):
            rfGraphCol, _, rfEvalCol= st.columns([10, 1, 6])

            with rfGraphCol:
                forestModel = getModel('rf_regression.pkl', RandomForestRegressor(n_estimators=100, random_state=42), 
                                       X_train=X_train, y_train=y_train)

                y_pred_forest = forestModel.predict(X_test)
                y_pred_forest.clip(0, y_pred_forest.max())
                forestPredDollars = np.expm1(y_pred_forest) # undo log transform

                sortedforestResults = pd.DataFrame({'Actual Values': yTestDollars, 
                                            'Predicted Values': forestPredDollars}
                                            ).sort_values('Actual Values').reset_index(drop=True)
                
                forestPlot = px.line(sortedforestResults, y=["Predicted Values", "Actual Values"], 
                                        color_discrete_map={'Actual Values': 'green', 'Predicted Values': 'lightblue'})
                forestPlot.update_layout(title="Random Forest Regression")
                st.plotly_chart(forestPlot)

            with rfEvalCol:
                forestR2 = r2_score(y_test, y_pred_forest)
                forestRMSE = np.sqrt(mean_squared_error(yTestDollars, forestPredDollars))

                st.markdown("##### Model Evaluation")
                st.markdown(f"**R<sup>2</sup>** = {forestR2:.4f}", unsafe_allow_html=True)
                st.markdown(f"**RMSE** = ${forestRMSE:.2f}", unsafe_allow_html=True)

                st.markdown("###### The first ensemble model, showing a near perfect R<sup>2</sup> with the best RMSE " \
                "found so far. This model does a great job at capturing the general trend of the data, and guesses are only " \
                "off by \\$22.91 on average, which is excellent for this dataset. It seems like having multiple trees guessing " \
                "eliminated a lot of the noise in the previous model, and expensive properties are being undervalued.", 
                unsafe_allow_html=True)

        with st.container(border=True):
            xgbGraphCol, _, xgbEvalCol= st.columns([10, 1, 6])

            with xgbGraphCol:
                xgbModel = getModel('xgb_regression.pkl', GradientBoostingRegressor(n_estimators=100, random_state=42), 
                                       X_train=X_train, y_train=y_train)

                y_pred_xgb = xgbModel.predict(X_test)
                y_pred_xgb.clip(0, y_pred_xgb.max())
                xgbPredDollars = np.expm1(y_pred_xgb) # undo log transform

                sortedxgbResults = pd.DataFrame({'Actual Values': yTestDollars, 
                                            'Predicted Values': xgbPredDollars}
                                            ).sort_values('Actual Values').reset_index(drop=True)
                
                xgbPlot = px.line(sortedxgbResults, y=["Predicted Values", "Actual Values"], 
                                        color_discrete_map={'Actual Values': 'green', 'Predicted Values': 'lightblue'})
                xgbPlot.update_layout(title="Gradient Boost Regression")
                st.plotly_chart(xgbPlot)

            with xgbEvalCol:
                xgbR2 = r2_score(y_test, y_pred_xgb)
                xgbRMSE = np.sqrt(mean_squared_error(yTestDollars, xgbPredDollars))

                st.markdown("##### Model Evaluation")
                st.markdown(f"**R<sup>2</sup>** = {xgbR2:.4f}", unsafe_allow_html=True)
                st.markdown(f"**RMSE** = ${xgbRMSE:.2f}", unsafe_allow_html=True)

                st.markdown("###### The second ensemble model. Similar performance to the random forest model, but with "
                "slightly worse performance in both metrics.",
                unsafe_allow_html=True)

    with significantFeaturesTab:
        st.subheader("Extracting Most Significant Features From RandomForest and XGBoost")

        impCol1, impCol2 = st.columns([1, 1])

        with impCol1:
            st.markdown("##### Random Forest")
            forestImportance = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': forestModel.feature_importances_
            }).sort_values('Importance', ascending=False)

            st.dataframe(forestImportance)

        with impCol2:
            st.markdown("##### XGBoost")
            xgbImportance = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': xgbModel.feature_importances_
            }).sort_values('Importance', ascending=False)

            st.dataframe(xgbImportance)

        st.markdown("For both of the best performing models, the most important features were **bedrooms**, "
        "**estimated_revenue_l365d**, **estimated_occupancy_l365d**, **bathrooms**, and **accomodates**. " \
        "The fact that **estimated_occupancy_l365d** served as a good predictor for price further supports my hypothesis.")