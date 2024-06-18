# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:59:18 2024

@author: Vitor Hugo Mourão & Natália dos Reis
"""

import os
import sys

# Ensure the script directory is added to the system path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# Optionally, change the working directory
os.chdir(project_root)

import pandas as pd
from utils.io_utils.py import load_data, save_data

def preprocess_data(filepath):
    df = load_data(filepath)
    # Add preprocessing steps here
    save_data(df, 'data/processed/train_processed.csv')