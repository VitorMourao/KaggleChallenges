# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:19:21 2024

@author: Vitor Hugo Mourão & Natália dos Reis
"""

from utils.clean_test import reduce_test_df

def women_first(train, test):
    """
    Assigns survival predictions based on the 'women and children first' principle.
    
    This function assumes that all female passengers survived and all male passengers did not survive.
    It adds a 'Survived' column to the provided DataFrame and assigns:
    - 1 for female passengers (indicating survival)
    - 0 for male passengers (indicating no survival)
    
    Parameters:
    train (pd.DataFrame): A DataFrame containing the training data with at least a 'Sex' and 'Survived' column.
    test (pd.DataFrame): A DataFrame containing the test data with at least a 'Sex' column.
    
    Returns:
    pd.DataFrame: The input DataFrame with an additional 'Survived' column filled with survival predictions.
    """
    
    # Predict survival for the training data
    train_df = train.copy()
    train_df['PredictedSurvived'] = 0  # Initialize all passengers as not survived
    train_df.loc[train_df['Sex'] == 'female', 'PredictedSurvived'] = 1  # Set 'PredictedSurvived' to 1 for female passengers
    
    # Calculate accuracy
    accuracy = (train_df['Survived'] == train_df['PredictedSurvived']).mean()
    print(f"Accuracy on training data: {accuracy * 100:.2f}%")
    
    # Predict survival for the test data
    test_df = test.copy()
    test_df['Survived'] = 0 # Initialize all passengers as not survived
    test_df.loc[test_df['Sex'] == 'female', 'Survived'] = 1 # Set 'Survived' to 1 for female passengers
    test_df = reduce_test_df(test_df)
    return test_df