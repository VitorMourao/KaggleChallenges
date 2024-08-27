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
from config import target, max_depth_range, number_of_trees
from utils.io_utils import load_data, save_data
from scripts.data_preprocessing import preprocess_data
from models.fem_survived_male_died import women_first
from models.logistic_regression import train_logistic_regression, test_logistic_regression
from models.decision_tree import train_decision_tree, test_decision_tree
from models.random_forest import train_random_forest, test_random_forest
from models.neural_network import train_neural_network, test_neural_network

def main(models, train_old, test_old, train_new, test_new):
    if 'model_01' in models:
        # Implement the first model (Only women survive)
        model_01_df = women_first(train_old, test_old) # Does not need any training
        model_01_df_data_path = os.path.join(test_predict_path, 'test_01.csv')
        save_data(model_01_df, model_01_df_data_path)
    
    if 'model_02' in models:
        # Implement Logistic Regression 
        model_02 = train_logistic_regression(train_new)
        model_02_df = test_logistic_regression(model_02, test_new)
        model_02_df_data_path = os.path.join(test_predict_path, 'test_02.csv')
        save_data(model_02_df, model_02_df_data_path)
    
    if 'model_03' in models:
        # Implement decision tree 
        model_03 = train_decision_tree(train_new, target, max_depth_range)
        model_03_df = test_decision_tree(model_03, test_new)
        model_03_df_data_path = os.path.join(test_predict_path, 'test_03.csv')
        save_data(model_03_df, model_03_df_data_path)
        
    
    if 'model_04' in models:
        # Implement Random Forest
        model_04 = train_random_forest(train_new, target, max_depth_range, number_of_trees)
        model_04_df = test_random_forest(model_04, test_new)
        model_04_df_data_path = os.path.join(test_predict_path, 'test_04.csv')
        save_data(model_04_df, model_04_df_data_path)
    
    if 'model_05' in models:
        model_05, preprocessor = train_neural_network(train_new, target)
        model_05_df = test_neural_network(test_new, target, model_05, preprocessor)
        model_05_df_data_path = os.path.join(test_predict_path, 'test_05.csv')
        save_data(model_05_df, model_05_df_data_path)
    

# If this script is being run as the main program
if __name__ == '__main__':
    # Old data
    train_old = load_data(train_data_path)
    test_old = load_data(test_data_path)
    
    # Preprocess the data
    preprocess_data(train_data_path, train_data_processed, **train_preprocessing_params)
    preprocess_data(test_data_path, test_data_processed, **test_preprocessing_params)
    train_new = load_data(train_data_processed)
    test_new = load_data(test_data_processed)
    
    # Prompt user for input
    available_models = ['model_01', 'model_02', 'model_03', 'model_04', 'model_05']
    print("Available models: ", available_models)
    models_to_run = input("Enter the models you want to run (separated by space): ").split()
    
    # Check if the user entered "0" to skip running any model
    if models_to_run == ['0']:
        print("No models selected. Exiting without running any models.")
        exit(0)
    
    # Validate input
    for model in models_to_run:
        if model not in available_models:
            print(f"Model {model} is not available. Please choose from {available_models}.")
            exit(1)

    main(models_to_run, train_old, test_old, train_new, test_new)
    