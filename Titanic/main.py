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
from config import train_data_path, train_data_processed, test_data_path, test_data_processed, test_predict_path, model_eval_path
from config import train_preprocessing_params, test_preprocessing_params
from utils.io_utils import load_data, save_data
from scripts.data_preprocessing import preprocess_data
from models.fem_survived_male_died import women_first
from models.logistic_regression import train_logistic_regression, test_logistic_regression

# If this script is being run as the main program
if __name__ == '__main__':
    train_old = load_data(train_data_path)
    preprocess_data(train_data_path, train_data_processed, **train_preprocessing_params) # preprocess the training data
    train_new = load_data(train_data_processed)
    
    # Save the test dataset
    test_old = load_data(test_data_path)
    preprocess_data(test_data_path, test_data_processed, **test_preprocessing_params)
    test_new = load_data(test_data_processed)

    # Implement the first model (Only women survive)
    model_01_df = women_first(train_old, test_old) # Does not need any training
    model_01_df_data_path = os.path.join(test_predict_path, 'test_01.csv')
    save_data(model_01_df, model_01_df_data_path)
    
    # Implement Logistic Regression 
    model_02 = train_logistic_regression(train_new)
    model_02_df = test_logistic_regression(model_02, test_new)
    model_02_df_data_path = os.path.join(test_predict_path, 'test_02.csv')
    save_data(model_02_df, model_02_df_data_path)
    