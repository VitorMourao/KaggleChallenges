# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 20:56:48 2024

@author: Vitor Hugo Mourão & Natália dos Reis
"""

import os

def get_root_path():
    return os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    print(get_root_path())