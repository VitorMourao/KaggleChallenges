# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 17:58:51 2024

@author: Vitor Hugo Mourão & Natália dos Reis
"""

import pandas as pd
from sklearn.impute import KNNImputer
import logging

def apply_knn_imputation(df, knn_columns, n_neighbors=5):
    """
    Apply KNN imputation to specified columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    knn_columns (list): List of columns to apply KNN imputation.
    n_neighbors (int): Number of neighbors to use for KNN imputation.

    Returns:
    pd.DataFrame: DataFrame with KNN imputation applied to specified columns.
    """
    numeric_columns = df.select_dtypes(include=['number']).columns
    imputer = KNNImputer(n_neighbors=n_neighbors)
    knn_df = df[numeric_columns].copy()
    imputed_knn_values = imputer.fit_transform(knn_df)
    imputed_knn_df = pd.DataFrame(imputed_knn_values, columns=numeric_columns)
    for col in knn_columns:
        df[col] = imputed_knn_df[col]
    logging.info(f"Applied KNNImputer with {n_neighbors} neighbors on columns: {knn_columns}.")
    
    return df