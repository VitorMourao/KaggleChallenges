# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:59:18 2024

@author: Vitor Hugo Mourão & Natália dos Reis
"""

from utils.io_utils import load_data, save_data
from sklearn.impute import KNNImputer

def missing_values(df):
    df.drop(columns=['Cabin'], inplace=True)  # Dropping 'Cabin' due to too many missing values
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True) # Only one missing value
    
    # Select numeric columns for KNNImputer (KNNImputer works on numeric data)
    numeric_columns = df.select_dtypes(include=['number']).columns
    
    # Apply KNNImputer on the entire dataset
    imputer = KNNImputer(n_neighbors=5)
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    return df

def preprocess_data(filepath_I,filepath_O):
    df = load_data(filepath_I)
    copy_df = df.copy()
    copy_df = missing_values(copy_df)
    save_data(copy_df, filepath_O)
    return




# # Verify missing values after cleaning
# print("\nMissing Values After Cleaning")
# print(df.isnull().sum())

# Balance classes