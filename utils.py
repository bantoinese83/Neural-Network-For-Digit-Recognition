# utils.py

import pandas as pd
import os
from tensorflow.keras.models import Model


def load_data(path):
    # Check if a path is a string
    if not isinstance(path, str):
        raise ValueError("Path should be a string")

    # Check if file exists at the given path
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No file found at {path}")

    # Check if file is a CSV file
    if not path.endswith('.csv'):
        raise ValueError("File should be a CSV file")

    # Load the data
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Failed to load data. Error: {str(e)}")
        return None

    return df


def save_data(df, path):
    # Check if a path is a string
    if not isinstance(path, str):
        raise ValueError("Path should be a string")

    # Check if the DataFrame is not None
    if df is None:
        raise ValueError("DataFrame is None. Cannot save None data")

    # Save the data
    try:
        df.to_csv(path, index=False)
        print(f"Data saved at {path}")
    except Exception as e:
        print(f"Failed to save data. Error: {str(e)}")


def save_model(model, path):
    # Check if a path is a string
    if not isinstance(path, str):
        raise ValueError("Path should be a string")

    # Check if the model is not None
    if model is None:
        raise ValueError("Model is None. Cannot save None model")

    # Check if the model is a Keras Model
    if not isinstance(model, Model):
        raise ValueError("Model should be a Keras Model")

    # Save the model
    try:
        model.save(path)
        print(f"Model saved at {path}")
    except Exception as e:
        print(f"Failed to save model. Error: {str(e)}")
