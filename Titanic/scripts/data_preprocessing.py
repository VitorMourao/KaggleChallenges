# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:59:18 2024

@author: Vitor Hugo Mourão & Natália dos Reis
"""

import pandas as pd
from utils.io_utils import load_data, save_data

def preprocess_data(filepath):
    df = load_data(filepath)
    # Add preprocessing steps here
    save_data(df, 'data/processed/train_processed.csv')
    
# Load the data using a path relative to the project root
try:
    data = load_data('data/raw/train.csv')
except Exception as e:
    print(f"Failed to load data: {e}")

# Process the data (this is just an example, replace with actual processing)
processed_data = data.copy()  # Replace this with actual processing steps

# Save the processed data using a path relative to the project root
try:
    save_data(processed_data, 'data/processed/processed_train.csv')
except Exception as e:
    print(f"Failed to save data: {e}")