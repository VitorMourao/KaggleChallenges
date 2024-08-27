# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:59:18 2024

@author: Vitor Hugo Mourão & Natália dos Reis
"""

from utils.io_utils import load_data, save_data
from utils.imputation_utils import apply_knn_imputation
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging
import re
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
        df = apply_knn_imputation(df, knn_columns, n_neighbors)
        
    return df

# Function to classify marital status as binary (1 for married, 0 for unmarried/unknown)
def get_marital_status_binary(name):
    # Extract the title using a regular expression
    title_search = re.search(r',\s*([^\.]*)\.', name)
    if title_search:
        title = title_search.group(1).strip()
        
        # Binary classification for marital status
        if title == 'Mrs':
            return 1  # Married
        else:
            return 0  # Unmarried or Unknown
    else:
        return 0  # Return 0 for unknown cases
    
# Function to apply binary marital status extraction on the entire DataFrame
def extract_marital_status_binary(df):
    # Apply the get_marital_status_binary function to the Name column and create a new binary column
    df['Married'] = df['Name'].apply(get_marital_status_binary)
    return df

# Function to count the number of names, excluding those in parentheses
def count_names_excluding_parentheses(name):
    # Remove content inside parentheses
    name_without_parentheses = re.sub(r'\(.*?\)', '', name).strip()
    # Split the name by spaces to count the number of parts (names)
    name_parts = name_without_parentheses.split()
    # Count the number of words that are actual names (ignoring titles like Mr, Mrs)
    return len([part for part in name_parts if part not in ['Mr.', 'Mrs.', 'Miss.', 'Master.', 'Dr.', 'Rev.']])

# Function to add the number of names column to the DataFrame
def add_name_count_column(df):
    # Apply the count_names_excluding_parentheses function to the Name column
    df['Name_Count'] = df['Name'].apply(count_names_excluding_parentheses)
    return df

def normalization(df, normalize_columns, method):
    """
    Normalize numeric columns
    
    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    normalize_columns (list): List of columns to normalize.
    method (str): String that decides which normalization methods will be used
                           String: 'mm', 'std'
                           Methods: 'MinMaxScaler', "StandardScaker".
                           These columns must be numeric.
    Returns:
    pd.DataFrame: DataFrame with normalized values.
    """
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

def preprocess_data(filepath_I,filepath_O, drop_columns, fillna_columns, n_neighbors, normalize_columns, method):
    df = load_data(filepath_I)
    copy_df = df.copy()
    copy_df = missing_values(copy_df, drop_columns, fillna_columns, n_neighbors)
    copy_df = extract_marital_status_binary(copy_df)
    copy_df = add_name_count_column(copy_df)
    copy_df = normalization(copy_df, normalize_columns, method)
    #TODO: Balance the target
    
    save_data(copy_df, filepath_O)
    return