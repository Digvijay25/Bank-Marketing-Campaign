import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set the style for plots
sns.set(style="whitegrid")
plt.style.use('seaborn-v0_8-whitegrid')

# Create directories for saving plots
os.makedirs('./visualizations/exploratory', exist_ok=True)

# Load the data
print("Loading the dataset...")
df = pd.read_csv('./data/bank.csv', sep=',')

# Basic information about the dataset
print("\n=== Dataset Information ===")
print(f"Dataset shape: {df.shape}")
print(f"Number of samples: {df.shape[0]}")
print(f"Number of features: {df.shape[1]}")

# Display the first few rows
print("\n=== First 5 rows of the dataset ===")
print(df.head())

# Data types and missing values
print("\n=== Data Types and Missing Values ===")
print(df.info())

# Summary statistics
print("\n=== Summary Statistics ===")
print(df.describe())

# Check for missing values
print("\n=== Missing Values ===")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values found")

# Check for 'unknown' values in categorical columns
print("\n=== 'Unknown' Values in Categorical Columns ===")
for col in df.select_dtypes(include=['object']).columns:
    unknown_count = df[df[col] == 'unknown'].shape[0]
    if unknown_count > 0:
        print(f"{col}: {unknown_count} unknown values ({unknown_count/df.shape[0]*100:.2f}%)")

# Target variable distribution
print("\n=== Target Variable Distribution ===")
target_counts = df['deposit'].value_counts()
print(target_counts)
print(f"Percentage of 'yes': {target_counts['yes']/df.shape[0]*100:.2f}%")
print(f"Percentage of 'no': {target_counts['no']/df.shape[0]*100:.2f}%")

# Save target distribution plot
plt.figure(figsize=(10, 6))
sns.countplot(x='deposit', data=df)
plt.title('Target Variable Distribution')
plt.savefig('./visualizations/exploratory/target_distribution.png')
plt.close()

# Analyze numerical features
print("\n=== Numerical Features Analysis ===")
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
print(f"Numerical features: {list(numerical_features)}")

# Distribution of numerical features
for feature in numerical_features:
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x='deposit', y=feature, data=df)
    plt.title(f'{feature} by Target')
    
    plt.tight_layout()
    plt.savefig(f'./visualizations/exploratory/{feature}_analysis.png')
    plt.close()

# Analyze categorical features
print("\n=== Categorical Features Analysis ===")
categorical_features = df.select_dtypes(include=['object']).columns
print(f"Categorical features: {list(categorical_features)}")

# Distribution of categorical features
for feature in categorical_features:
    if feature != 'y':  # Skip the target variable
        plt.figure(figsize=(12, 6))
        
        # Count plot
        plt.subplot(1, 2, 1)
        value_counts = df[feature].value_counts()
        sns.countplot(y=feature, data=df, order=value_counts.index)
        plt.title(f'Distribution of {feature}')
        plt.tight_layout()
        
        # Relationship with target
        plt.subplot(1, 2, 2)
        pd.crosstab(df[feature], df['deposit'], normalize='index').plot(kind='bar', stacked=True)
        plt.title(f'{feature} vs Target')
        plt.ylabel('Proportion')
        
        plt.tight_layout()
        plt.savefig(f'./visualizations/exploratory/{feature}_analysis.png')
        plt.close()

# Correlation analysis for numerical features
print("\n=== Correlation Analysis ===")
correlation_matrix = df.select_dtypes(include=['int64', 'float64']).corr()
print(correlation_matrix)

# Save correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('./visualizations/exploratory/correlation_heatmap.png')
plt.close()

print("\nData exploration completed. Visualizations saved in './visualizations/exploratory/' directory.")
