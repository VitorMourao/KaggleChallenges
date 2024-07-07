# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 16:59:47 2024

@author: Vitor Hugo Mourão & Natália dos Reis
"""

import os

# Paths
train_data_path = os.path.join('data', 'raw', 'train.csv')
test_data_path = os.path.join('data', 'raw', 'test.csv')
train_data_processed = os.path.join('data', 'processed', 'train_processed.csv')
test_data_processed = os.path.join('data', 'processed', 'test_processed.csv')
test_predict_path = os.path.join('outputs', 'test_predictions')
model_eval_path = os.path.join('outputs', 'model_evaluations')

# Preprocessing parameters
train_preprocessing_params = {
    'drop_columns': ['Cabin'],
    'fillna_columns': {'Embarked': 'mode', 'Age': 'knn'},
    'n_neighbors': 5,
    'normalize_columns': ['Age', 'SibSp', 'Parch', 'Fare'],
    'method': 'std'
}

test_preprocessing_params = {
    'drop_columns': ['Cabin'],
    'fillna_columns': {'Fare': 'mean', 'Age': 'knn'},
    'n_neighbors': 5,
    'normalize_columns': ['Age', 'SibSp', 'Parch', 'Fare'],
    'method': 'std'
}