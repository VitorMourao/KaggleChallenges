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