# while it doesn't follow the flow of data, outliers are being removed at the end of the
# data cleaning file instead of on the actual EDA tab. However, to initialize the app
# with the finalized dataset, it's being done here, then passed to the EDA tab where we
# pretend that's where it's being done.

import pandas as pd

def removeOutliers(df: pd.DataFrame):
    upper = df['price'].quantile(0.998)
    clippedPriceDf = df[df['price'] <= upper]
    finalDf = clippedPriceDf[clippedPriceDf['minimum_nights'] <= 31]

    return finalDf