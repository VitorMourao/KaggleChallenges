# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:19:21 2024

@author: Vitor Hugo Mourão & Natália dos Reis
"""

def women_first(test_df):
    """
    Assigns survival predictions based on the 'women and children first' principle.
    
    This function assumes that all female passengers survived and all male passengers did not survive.
    It adds a 'Survived' column to the provided DataFrame and assigns:
    - 1 for female passengers (indicating survival)
    - 0 for male passengers (indicating no survival)
    
    Parameters:
    test_df (pd.DataFrame): A DataFrame containing the test data with at least a 'Sex' column.
    
    Returns:
    pd.DataFrame: The input DataFrame with an additional 'Survived' column filled with survival predictions.
    """
    test_df['Survived'] = 0 # Initialize all passengers as not survived
    test_df.loc[test_df['Sex'] == 'female', 'Survived'] = 1 # Set 'Survived' to 1 for female passengers
    return test_df