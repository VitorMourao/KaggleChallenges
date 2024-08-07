# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 20:00:28 2024

@author: Vitor Hugo Mourão & Natália dos Reis

"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_logistic_regression(train_data):
    """
   Trains a logistic regression model on the provided training data and evaluates its accuracy.

   Parameters:
   train_data (pd.DataFrame): A DataFrame containing the training data with a 'Survived' column 
                              indicating the target variable and a 'PassengerId' column to be excluded.

   Returns:
   LogisticRegression: The trained logistic regression model.
   """
    # Create a copy of the training data
    data = train_data.copy()

    # Select only numeric columns and exclude 'PassengerId' and 'Survived'
    X = data.select_dtypes(include=[float, int]).drop(columns=['PassengerId', 'Survived'])
    y = data['Survived']

    # Initialize and train the logistic regression model
    model = LogisticRegression(random_state=42)
    model.fit(X, y)

    # Predict on the training data to check accuracy
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)*100
    print(f'Accuracy on training data (Model_02): {accuracy:.2f}%')

    return model

def test_logistic_regression(model, test_data):
    """
    Tests a logistic regression model on the provided test data and generates survival predictions.

    Parameters:
    model (LogisticRegression): The trained logistic regression model.
    test_data (pd.DataFrame): A DataFrame containing the test data with a 'PassengerId' column.

    Returns:
    pd.DataFrame: A DataFrame containing 'PassengerId' and the predicted 'Survived' values.
    """
    # Create a copy of the test data
    data = test_data.copy()

    # Select only numeric columns and exclude 'PassengerId'
    X_test = data.select_dtypes(include=[float, int]).drop(columns=['PassengerId'])

    # Ensure all columns match between training and test datasets
    missing_cols = set(model.feature_names_in_) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0
    X_test = X_test[model.feature_names_in_]

    # Make predictions
    predictions = model.predict(X_test)

    # Create the output DataFrame
    output_df = pd.DataFrame({
        data.columns[0]: data.iloc[:, 0],
        'Survived': predictions
    })

    return output_df
