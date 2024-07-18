# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 20:36:28 2024

@author: Vitor Hugo Mourão & Natália dos Reis
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.callbacks import EarlyStopping

def train_neural_network(train_data, target):
    # Create a copy of the training data
    data = train_data.copy()
    
    # Separate the target variable from the features
    X = data.drop(columns=[target])
    y = data[target]
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=['number']).columns
    
    # Define the column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    
    # Preprocess the features
    X = preprocessor.fit_transform(X)
    
    # Ensure the target is numeric
    y = y.astype(np.float32)
    
    # Define the neural network model
    model = Sequential()
    model.add(Dense(8, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Define early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Can also use 'val_accuracy'
        patience=10,  # Number of epochs to wait for improvement
        restore_best_weights=True,  # Restore the weights of the best epoch
        min_delta=0.001,  # Minimum change to qualify as an improvement
        verbose=1
    )
    
    # Train the model with early stopping
    model.fit(
        X, y, 
        epochs=1000, 
        batch_size=32, 
        verbose=1, 
        validation_split=0.1,  # Use % of the data for validation
        callbacks=[early_stopping]
    )
    
    return model, preprocessor

def test_neural_network(test_data, target, model, preprocessor):
    
    # Create a copy of the training data
    data = test_data.copy()
    
    # Preprocess the features using the same preprocessor from training
    X_test = preprocessor.transform(data)
    
    # Convert X_test to numpy array
    X_test = X_test.astype(np.float32)
    
    # Make predictions
    predictions = model.predict(X_test)
    predicted_classes = (predictions > 0.5).astype(int).flatten()
    
    # Create output DataFrame
    output_df = data.iloc[:, [0]].copy()  # Assuming the first column is the identifier
    output_df[target] = predicted_classes
    
    return output_df