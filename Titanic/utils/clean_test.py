# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:36:29 2024

@author: vitor
"""

def reduce_test_df(test_df):
    """
    Reduces the DataFrame to include only 'PassengerId' and 'Survived' columns.
    
    Parameters:
    test_df (pd.DataFrame): The input DataFrame containing at least 'PassengerId' and 'Survived' columns.
    
    Returns:
    pd.DataFrame: A new DataFrame containing only 'PassengerId' and 'Survived' columns.
    """
    # Select only 'PassengerId' and 'Survived' columns
    new_df = test_df[['PassengerId', 'Survived']]
    return new_df