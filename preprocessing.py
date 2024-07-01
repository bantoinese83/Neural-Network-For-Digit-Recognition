# preprocessing.py
import pandas as pd


def preprocess_data(df):
    # Check if df is a pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input should be a pandas DataFrame")

    # Check if 'label' column exists in df
    if 'label' in df.columns:
        y = df['label']
        X = df.drop(columns=['label'])
    else:
        X = df
        y = None

    # Check if the DataFrame is not empty
    if X.empty:
        raise ValueError("DataFrame should not be empty")

    # Normalize pixel values to [0, 1] range
    X = X / 255.0

    # Check if any NaN values and fill them with 0
    if X.isnull().values.any():
        X.fillna(0, inplace=True)

    return X, y
