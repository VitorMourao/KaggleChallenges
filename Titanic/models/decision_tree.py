# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 22:53:31 2024

@author: Vitor Hugo Mourão & Natália dos Reis
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn import tree
import matplotlib.pyplot as plt

def plot_the_tree(feature_names, class_names, model):
    """
    D-Tree Plot
    """
    # Plot the decision tree
    plt.figure(figsize=(20,10))
    tree.plot_tree(model, filled=True, feature_names=feature_names, class_names=class_names)
    plt.show()
    
def perform_cross_validation(X, y, max_depth_range):
    """
    Perform cross-validation to find the best max depth.
    """
    cv_scores = {}
    
    for max_depth in max_depth_range:
        # Create decision tree classifier
        clf = DecisionTreeClassifier(max_depth=max_depth)
        
        # Perform 10-fold cross-validation
        scores = cross_val_score(clf, X, y, cv=10)
        
        # Calculate the mean score and store it in the dictionary
        cv_scores[max_depth] = np.mean(scores)
        print(f"Max depth: {max_depth}, Cross-validation accuracy: {cv_scores[max_depth]}")
    
    return cv_scores

def train_decision_tree(train_data, target, max_depth_range):
    """
    D-Tree Train
    """
    try:
        # Create a copy of the training data
        data = train_data.copy()
        
        # Separate the target variable from the features
        X = data.drop(columns=[target])
        y = data[target]
        
        # Convert categorical features to numeric
        X = pd.get_dummies(X)
        
        # Perform cross-validation
        cv_scores = perform_cross_validation(X, y, max_depth_range)
        
        # Find the max_depth with the highest cross-validation score
        best_max_depth = max(cv_scores, key=cv_scores.get)
        print(f"Best max depth based on cross-validation: {best_max_depth}")
        
        # Train the final model using the best max_depth
        final_clf = DecisionTreeClassifier(max_depth=best_max_depth)
        final_clf.fit(X, y)
        
        # Assess the accuracy on the training data
        accuracy = final_clf.score(X, y)
        print(f"Accuracy on training data (Model_03): {accuracy}")
        
        # Plot the tree
        feature_names = X.columns.tolist()
        class_names = y.unique().astype(str).tolist()
        plot_the_tree(feature_names, class_names, final_clf)
        
        return final_clf
    except Exception as e:
        print(f"An error occurred: {e}")

def test_decision_tree(model, test_data):
    """
    D-Tree Test
    """
    # Create a copy of the training data
    data = test_data.copy()
    
    # Convert categorical features to numeric (same as training)
    X_test = pd.get_dummies(data)
    
    # Predict the target values for the test data
    predictions = model.predict(X_test)
    
    # Create the output DataFrame
    output_df = pd.DataFrame({
        test_data.columns[0]: test_data.iloc[:, 0],
        'Survived': predictions
    })

    return output_df