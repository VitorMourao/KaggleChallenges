# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 20:36:28 2024

@author: Vitor Hugo Mourão & Natália dos Reis
"""

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def train_neural_network(train_data, target):
    # Create a copy of the training data
    data = train_data.copy()
    
    # Separate the target variable from the features
    X = data.drop(columns=[target])
    y = data[target]
    
    # Convert categorical features to numeric
    X = pd.get_dummies(X)
    
    # Define the neural network model
    model = Sequential()
    model.add(Dense(16, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X, y, epochs=50, batch_size=32, verbose=1)
    
    return model

def test_neural_network(test_data, target, model):
    
    # Create a copy of the training data
    data = test_data.copy()
    
    # Convert categorical features to numeric (same as training)
    X_test = pd.get_dummies(data)
    
    # Make predictions
    predictions = model.predict(X_test)
    predicted_classes = (predictions > 0.5).astype(int).flatten()
    
    # Create output DataFrame
    output_df = test_data.iloc[:, [0]].copy()  # Assuming the first column is the identifier
    output_df['Survived'] = predicted_classes
    
    output_df = 1
    return output_df