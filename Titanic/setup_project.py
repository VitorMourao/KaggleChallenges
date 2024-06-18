# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 18:07:28 2024

@author: Vitor Hugo Mourão & Natália dos Reis
"""

import os
import sys
import subprocess

def add_paths(root_dir):
    """
    Recursively add all directories under root_dir to sys.path
    """
    for dirpath, _, _ in os.walk(root_dir):
        if dirpath not in sys.path:
            sys.path.append(dirpath)
            print(f"Added {dirpath} to sys.path")

def install_dependencies(root_dir):
    """
    Recursively find and install dependencies listed in all requirements.txt files under root_dir
    """
    for dirpath, _, filenames in os.walk(root_dir):
        if 'requirements.txt' in filenames:
            requirements_path = os.path.join(dirpath, 'requirements.txt')
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_path])
            print(f"Installed dependencies from {requirements_path}")

def main():
    project_root = os.path.abspath(os.path.dirname(__file__))
    
    print("Adding paths to sys.path...")
    add_paths(project_root)
    
    print("Installing dependencies...")
    install_dependencies(project_root)
    
    print("Project setup complete")

if __name__ == "__main__":
    main()
