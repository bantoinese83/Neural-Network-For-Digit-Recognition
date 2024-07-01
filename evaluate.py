# evaluate.py
import pandas as pd
import numpy as np

from preprocessing import preprocess_data
from tensorflow.keras.models import Model


def evaluate_model(df, model):
    # Check if df is a pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input should be a pandas DataFrame")

    # Check if model is a Keras Model
    if not isinstance(model, Model):
        raise ValueError("Model should be a Keras Model")

    # Preprocess the data
    X_test, y_test = preprocess_data(df)

    # Reshape the data
    X_test = X_test.values.reshape(-1, 28, 28, 1)

    # Print the shape of X_test and y_test after preprocessing
    print(f"Shape of X_test after preprocessing: {X_test.shape}")
    if y_test is not None:
        print(f"Shape of y_test after preprocessing: {y_test.shape}")

    # Check if the DataFrame is not empty
    if X_test.size == 0:
        raise ValueError("DataFrame should not be empty")

    # Make predictions if y_test is None (i.e., labels are not available)
    if y_test is None:
        predictions = model.predict(X_test)
        predicted_labels = np.argmax(predictions, axis=1)
        print(f"Predicted labels: {predicted_labels}")
    else:
        # Evaluate the model if y_test is not None (i.e., labels are available)
        try:
            test_loss, test_acc = model.evaluate(X_test, y_test)
            print(f"Test accuracy: {test_acc}")
        except Exception as e:
            print(f"Failed to evaluate the model. Error: {str(e)}")
