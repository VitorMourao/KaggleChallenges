# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 21:19:11 2024

@author: Vitor Hugo Mourão & Natália dos Reis
"""

# Importing the get_ipython function to access the IPython instance
from IPython import get_ipython
ipython = get_ipython()  # Getting the current IPython instance
ipython.run_line_magic('reload_ext', 'autoreload') # Reloading the 'autoreload' extension to ensure it is active
ipython.run_line_magic('autoreload', '2') # Setting 'autoreload' to mode 2, reloads all modules every time before executing the code

# Importing necessary modules
import os
import pandas as pd
from utils.io_utils import save_data
from utils.clean_test import reduce_test_df
from scripts.data_preprocessing import preprocess_data
from models.fem_survived_male_died import women_first

# Defining the paths to the training and test data files
train_data_path = os.path.join('data', 'raw', 'train.csv')
test_data_path = os.path.join('data', 'raw', 'test.csv')

# Defining the path to the processed training data file
train_data_processed = os.path.join('data', 'processed', 'train_processed.csv')

# If this script is being run as the main program
if __name__ == '__main__':
    preprocess_data(train_data_path, train_data_processed) # preprocess the training data
    data_old = pd.read_csv(train_data_path)
    data_new = pd.read_csv(train_data_processed)
    
    # Save the test dataset
    test_df = pd.read_csv(test_data_path)

    # Implement the first model (Only women survive)
    model_01_df = women_first(test_df) # Does not need any training
    model_01_df = reduce_test_df(model_01_df)
    model_01_df_data_path = os.path.join('outputs', 'test_predictions', 'test_01.csv')
    save_data(model_01_df, model_01_df_data_path)
    