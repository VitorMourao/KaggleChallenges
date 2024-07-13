# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 22:53:31 2024

@author: Vitor Hugo Mourão & Natália dos Reis
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
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

def train_decision_tree(train_data, target, max_depth=None):
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
    
        # Create decision tree classifier
        clf = DecisionTreeClassifier(max_depth=max_depth)
        
        # Train the classifier
        clf.fit(X, y)
        
        # Assess the accuracy on the training data
        accuracy = clf.score(X, y)
        print(f"Accuracy on training data (Model_03): {accuracy}")
        
        # Plot the tree
        feature_names = X.columns.tolist()
        class_names = y.unique().astype(str).tolist()
        plot_the_tree(feature_names, class_names, clf)
        
        return clf
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