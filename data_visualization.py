# data_visualization.py

import pandas as pd
from matplotlib import pyplot as plt


def visualize_data(df):
    # Check if df is a pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input should be a pandas DataFrame")

    # Check if 'label' column exists in df
    if 'label' not in df.columns:
        raise ValueError("'label' column not found in DataFrame")

    # Extract labels
    labels = df['label']

    # Check if the DataFrame is not empty
    if labels.empty:
        raise ValueError("DataFrame should not be empty")

    # Plot class distribution
    try:
        plt.figure(figsize=(10, 5))
        labels.value_counts().sort_index().plot(kind='bar')
        plt.xlabel('Class')
        plt.ylabel('Frequency')
        plt.title('Class Distribution')
        plt.show()
    except Exception as e:
        print(f"Failed to plot class distribution. Error: {str(e)}")

    # Display sample images
    try:
        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.imshow(df.iloc[i, 1:].values.reshape(28, 28), cmap='gray')
            plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Failed to display sample images. Error: {str(e)}")
