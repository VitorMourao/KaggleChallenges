# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 16:24:26 2024

@author: Vitor Hugo Mourão & Natália dos Reis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# To data input/output - Preprocessing
filepath_train = "train.csv"
filepaht_test = "test.csv"
df = pd.read_csv(filepath_train)
test = pd.read_csv(filepaht_test)

# Exploratory analysis of train
print("Basic Information")
print(df.info())

# Display the first few rows of the dataset
print("\nFirst 5 Rows")
print(df.head())

# Display summary statistics
print("\nSummary Statistics")
print(df.describe())

# Check for missing values
print("\nMissing Values")
print(df.isnull().sum())

# Visualize missing values
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

# Exploratory Data Analysis
# 1. Survival rate
print("\nSurvival Rate")
print(df['Survived'].value_counts(normalize=True))

# Plot survival count
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.show()

# 2. Survival rate by gender
print("\nSurvival Rate by Gender")
print(df.groupby('Sex')['Survived'].mean())

# Plot survival rate by gender
plt.figure(figsize=(6, 4))
sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Survival Rate by Gender')
plt.show()

# 3. Survival rate by class
print("\nSurvival Rate by Class")
print(df.groupby('Pclass')['Survived'].mean())

# Plot survival rate by class
plt.figure(figsize=(6, 4))
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Survival Rate by Class')
plt.show()

# 4. Age distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.show()

# 5. Survival rate by age with stacked bars
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Age', hue='Survived', multiple='stack', bins=30, kde=False, palette={0: 'red', 1: 'green'})
plt.title('Age Distribution by Survival')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# 5. Interactive violin plot for survival rate by age
fig = px.violin(df, y='Age', x='Survived', color='Survived', box=True, points='all',
                color_discrete_map={0: 'red', 1: 'green'},
                labels={'Survived': 'Survival Status'},
                title='Age Distribution by Survival')

fig.update_traces(hoverinfo='all', hovertemplate='Survival Status: %{x}<br>Age: %{y}')

# Save the figure to an HTML file
fig.write_html('age_distribution_by_survival_violin.html')

# 6. Correlation matrix
# Select only numeric columns
numeric_df = df.select_dtypes(include=['number'])
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
### CfS: Correlation with the target ###

# 8. Survival rate by family size
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
print("\nSurvival Rate by Family Size")
print(df.groupby('FamilySize')['Survived'].mean())
# Count how many people on each family

# Plot survival rate by family size
plt.figure(figsize=(10, 6))
sns.barplot(x='FamilySize', y='Survived', data=df, ci=None)
plt.title('Survival Rate by Family Size')
plt.show()

# 9. Pairplot for a quick overview
# Select new variables
sns.pairplot(df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked']], hue='Survived', diag_kind='kde')
plt.show()

# # PRE-PROCESSING: Data Cleaning: Fill missing values
# df['Age'].fillna(df['Age'].median(), inplace=True)
# df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
# df['Fare'].fillna(df['Fare'].median(), inplace=True)
# df.drop(columns=['Cabin'], inplace=True)  # Dropping 'Cabin' due to too many missing values

# # Verify missing values after cleaning
# print("\nMissing Values After Cleaning")
# print(df.isnull().sum())

# Balance classes