# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:46:38 2024

@author: Vitor Hugo Mourão & Natália dos Reis
"""

import pandas as pd

def load_data(filepath):
    return pd.read_csv(filepath)

def save_data(df, filepath):
    df.to_csv(filepath, index=False)