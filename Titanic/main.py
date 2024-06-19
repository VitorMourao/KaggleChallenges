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
from get_root_path import get_root_path
from scripts.data_preprocessing import preprocess_data

# Getting the root path of the project
root_path = get_root_path()

# Defining the paths to the training and test data files
train_data_path = os.path.join(root_path, 'data', 'raw', 'train.csv')
test_data_path = os.path.join(root_path, 'data', 'raw', 'test.csv')

# Defining the path to the processed training data file
train_data_processed = os.path.join(root_path, 'data', 'processed', 'train_processed.csv')

# If this script is being run as the main program, preprocess the training data
if __name__ == '__main__':
    preprocess_data(train_data_path, train_data_processed)
