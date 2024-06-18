# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:46:38 2024

@author: Vitor Hugo Mourão & Natália dos Reis
"""

import os
import pandas as pd
import inspect

def get_project_root():
    """Return the absolute path to the project root directory."""
    try:
        current_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
        project_root = os.path.join(current_file_path, os.pardir, os.pardir)
        return os.path.abspath(project_root)
    except Exception as e:
        print(f"Error getting project root: {e}")
        raise

def load_data(filepath):
    """Load a CSV file from a path relative to the project root."""
    try:
        project_root = get_project_root()
        full_path = os.path.join(project_root, filepath)
        print(f"Loading data from: {full_path}")
        return pd.read_csv(full_path)
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        raise

def save_data(df, filepath):
    """Save a DataFrame to a CSV file at a path relative to the project root."""
    try:
        project_root = get_project_root()
        full_path = os.path.join(project_root, filepath)
        print(f"Saving data to: {full_path}")
        df.to_csv(full_path, index=False)
    except Exception as e:
        print(f"Error saving data to {filepath}: {e}")
        raise
