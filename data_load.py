# data_load.py
import numpy as np
import pandas as pd
from sklearn import preprocessing
from utils import load_data


class DataLoader:
    def __init__(self, path):
        self.df = load_data(path)
        pd.set_option('display.max_rows', 20)
        pd.set_option('display.max_columns', 50)

    def display_data(self):
        print(self.df.head(500))
        print(self.df.tail(500))
        print(self.df.shape)

    def display_column_info(self):
        print(self.df.columns)
        print(self.df.dtypes)

    def display_missing_values(self):
        print(self.df.isnull().sum())

    def display_unique_values(self):
        print(self.df.nunique())
        if 'label' in self.df.columns:
            print(self.df['label'].nunique())
            print(self.df['label'].value_counts())

    def display_statistics(self):
        print(self.df.describe())
        print(self.df.corr())

    def remove_duplicates(self):
        print(self.df.duplicated().sum())
        self.df.drop_duplicates(inplace=True)
        print(self.df.shape)

    def fill_missing_values(self):
        # Fill missing values in the DataFrame
        self.df.fillna(self.df.mean(), inplace=True)

    def normalize_features(self):
        # Normalize numerical columns in the DataFrame
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            self.df[col] = preprocessing.MinMaxScaler().fit_transform(self.df[[col]])

    def encode_categorical_features(self):
        # Encode categorical columns in the DataFrame
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            self.df[col] = preprocessing.LabelEncoder().fit_transform(self.df[col])

    def remove_outliers(self):
        # Remove outliers from the DataFrame
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]


