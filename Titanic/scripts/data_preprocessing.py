# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:59:18 2024

@author: Vitor Hugo Mourão & Natália dos Reis
"""

from utils.io_utils import load_data, save_data
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging
import pandas as pd

def missing_values(df, drop_columns, fillna_columns, n_neighbors = 5):
    """
    Handle missing values in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    drop_columns (list): List of columns to drop due to too many missing values.
    fillna_columns (dict): Dictionary of columns to fill with their specified method.
                           Format: {'column_name': 'method'}
                           Methods: 'mode', 'mean', 'median', 'knn'.
    n_neighbors (int): Number of neighbors to use for KNN imputation.

    Returns:
    pd.DataFrame: DataFrame with missing values handled.
    """
    # Drop specified columns
    for col in drop_columns:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            logging.info(f"Column '{col}' dropped due to too many missing values.")
        else:
            logging.warning(f"Column '{col}' not found in DataFrame.")
            
    # Fill specified columns with their respective methods
    knn_columns = []
    for col, method in fillna_columns.items():
        if col in df.columns:
            if method == 'mode':
                fill_value = df[col].mode()[0]
                df[col].fillna(fill_value, inplace=True)
            elif method == 'mean':
                fill_value = df[col].mean()
                df[col].fillna(fill_value, inplace=True)
            elif method == 'median':
                fill_value = df[col].median()
                df[col].fillna(fill_value, inplace=True)
            elif method == 'knn':
                knn_columns.append(col)
                logging.info(f"Column '{col}' marked for KNN imputation.")
            else:
                logging.error(f"Unknown method '{method}' for column '{col}'.")
            logging.info(f"Column '{col}' filled using method '{method}' with value '{fill_value if method != 'knn' else 'KNN'}'.")
        else:
            logging.warning(f"Column '{col}' not found in DataFrame.")

    # Apply KNNImputer only to specified columns
    if knn_columns:
        numeric_columns = df.select_dtypes(include=['number']).columns
        imputer = KNNImputer(n_neighbors=n_neighbors)
        knn_df = df[numeric_columns].copy()
        imputed_knn_values = imputer.fit_transform(knn_df)
        imputed_knn_df = pd.DataFrame(imputed_knn_values, columns=numeric_columns)
        for col in knn_columns:
            df[col] = imputed_knn_df[col]
        logging.info(f"Applied KNNImputer with {n_neighbors} neighbors on columns: {knn_columns}.")
        
    return df

def normalization(df, normalize_columns, method):
    # Check if all columns in normalize_columns are numeric
    non_numeric_columns = [col for col in normalize_columns if not pd.api.types.is_numeric_dtype(df[col])]
    
    if non_numeric_columns:
        raise ValueError(f"Columns {non_numeric_columns} are not numeric and cannot be normalized.")
    
    if method == 'mm':
        scaler = MinMaxScaler()
    elif method == 'std':
        scaler = StandardScaler()
    else:
        raise ValueError("Method must be 'mm' (Min-Max Scaling) or 'std' (Standardization)")
    
    df[normalize_columns] = scaler.fit_transform(df[normalize_columns])
    return df

def preprocess_data(filepath_I,filepath_O):
    df = load_data(filepath_I)
    copy_df = df.copy()
    copy_df = missing_values(copy_df, drop_columns = ['Cabin'], fillna_columns={'Embarked':'mode', 'Age':'knn'}, n_neighbors=5)
    copy_df = normalization(copy_df, normalize_columns = ['Age','SibSp','Parch','Fare'], method = 'std')
    
    save_data(copy_df, filepath_O)
    return