# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 21:45:20 2024

@author: Vitor Hugo Mourão & Natália dos Reis
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

def perform_cross_validation_rf(X, y, max_depth_range):
    """
    Perform cross-validation to determine the optimal maximum depth for a Random forest Classifier.

    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        The input features for the model. This is the training data used for cross-validation.
        
    y : array-like of shape (n_samples,)
        The target values (class labels) corresponding to the input features.
        
    max_depth_range : list or array-like of int
        A list or range of integers representing the different maximum depths to evaluate 
        for the random forest classifier.

    Returns:
    --------
    cv_scores : dict
        A dictionary where the keys are the maximum depth values and the values are the mean 
        cross-validation accuracy scores for those depths.
    """
    cv_scores = {}
    
    for max_depth in max_depth_range:
        # Create decision tree classifier
        clf = RandomForestClassifier(max_depth=max_depth)
        
        # Perform 10-fold cross-validation
        scores = cross_val_score(clf, X, y, cv=10)
        
        # Calculate the mean score and store it in the dictionary
        cv_scores[max_depth] = np.mean(scores)
        print(f"Max depth: {max_depth}, Cross-validation accuracy: {cv_scores[max_depth]}")
    
    return cv_scores

def train_random_forest(train_data, target, max_depth_range, number_of_trees):
    
    # Create a copy of the training data
    data = train_data.copy()
    
    # Separate the target variable from the features
    X = data.drop(columns=[target])
    y = data[target]
    
    # Convert categorical features to numeric
    X = pd.get_dummies(X)
    
    # Perform cross-validation
    cv_scores = perform_cross_validation_rf(X, y, max_depth_range)
    
    # Find the max_depth with the highest cross-validation score
    best_max_depth = max(cv_scores, key=cv_scores.get)
    print(f"Best max depth based on cross-validation: {best_max_depth}")
    
    # Initialize the Random Forest classifier with given max_depth and number_of_tree
    rf_classifier = RandomForestClassifier(max_depth=best_max_depth, n_estimators=number_of_trees)
    
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