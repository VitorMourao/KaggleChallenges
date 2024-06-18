# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 18:07:28 2024

@author: Vitor Hugo Mourão & Natália dos Reis
"""

import os
import sys
import subprocess

def add_paths():
    # Add the parent directory of root to the system path
    project_root = os.path.abspath(os.path.dirname(__file__))
    if project_root not in sys.path:
        sys.path.append(project_root)
        print(f"Added {project_root} to sys.path")

def install_dependencies():
    # Install dependencies from requirements.txt
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_path])
        print("Dependencies installed")
    else:
        print("requirements.txt not found")

def main():
    add_paths()
    install_dependencies()
    print("Project setup complete")

if __name__ == "__main__":
    main()