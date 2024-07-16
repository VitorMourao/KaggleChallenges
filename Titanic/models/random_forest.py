# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 21:45:20 2024

@author: Vitor Hugo Mourão & Natália dos Reis
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

def train_random_forest(train_data, target, max_depth, number_of_trees):
    
    # Create a copy of the training data
    data = train_data.copy()
    
    # Separate the target variable from the features
    X = data.drop(columns=[target])
    y = data[target]
    
    # Convert categorical features to numeric
    X = pd.get_dummies(X)
    
    # Initialize the Random Forest classifier with given max_depth and number_of_tree
    rf_classifier = RandomForestClassifier(max_depth=max_depth, n_estimators=number_of_trees)
    
    # Fit the model with the training data
    rf_classifier.fit(X,y)
    
    # Predict on the training data
    train_predictions = rf_classifier.predict(X)
    
    # Calculate accuracy on the training data
    train_accuracy = accuracy_score(y, train_predictions)
    
    # Print the training accuracy
    print(f'Accuracy on training data: {train_accuracy:.4f}')
    
    # Return the trained model
    return rf_classifier

def test_random_forest(model, test_data):
    # Create a copy of the test data
    data = test_data.copy()
    
    # Store the first column of the test data
    first_column = data.iloc[:, 0]
    
    # Convert categorical features to numeric
    X_test = pd.get_dummies(data)
    
    # Predict on the test data
    test_predictions = model.predict(X_test)
    
    # Create a DataFrame with the first column and the predictions
    results_df = pd.DataFrame({test_data.columns[0]: first_column, 'Survived': test_predictions})
    
    # Return the results DataFrame
    return results_df