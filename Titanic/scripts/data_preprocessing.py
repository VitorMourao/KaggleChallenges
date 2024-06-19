# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:59:18 2024

@author: Vitor Hugo Mourão & Natália dos Reis
"""

from utils.io_utils import load_data, save_data

def preprocess_data(filepath_I,filepath_O):
    df = load_data(filepath_I)
    copy_df = df.copy()
    #Preprocess code...
    save_data(copy_df, filepath_O)
    return