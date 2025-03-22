import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# Set the style for plots
sns.set(style="whitegrid")
plt.style.use('seaborn-v0_8-whitegrid')

# Create directories for saving plots
os.makedirs('./visualizations/cleaned', exist_ok=True)

# Load the data
print("Loading the dataset...")
df = pd.read_csv('./data/bank.csv', sep=',')

print("\n=== Original Dataset ===")
print(f"Shape: {df.shape}")

# Make a copy of the original data
df_original = df.copy()

# 1. Handle 'unknown' values in categorical columns
print("\n=== Handling 'unknown' values ===")
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('deposit')  # Remove target variable

for col in categorical_cols:
    unknown_count = df[df[col] == 'unknown'].shape[0]
    if unknown_count > 0:
        print(f"{col}: {unknown_count} unknown values ({unknown_count/df.shape[0]*100:.2f}%)")
        
        # For columns with high percentage of unknowns (like poutcome), keep as 'unknown'
        # For columns with low percentage, replace with mode
        if unknown_count/df.shape[0] < 0.10:  # Less than 10% unknowns
            mode_value = df[df[col] != 'unknown'][col].mode()[0]
            df[col] = df[col].replace('unknown', mode_value)
            print(f"  - Replaced with mode: {mode_value}")
        else:
            print(f"  - Kept 'unknown' as a separate category")

# 2. Handle outliers in numerical columns
print("\n=== Handling outliers ===")
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

for col in numerical_cols:
    # Calculate IQR
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Count outliers
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_count = outliers.shape[0]
    
    if outlier_count > 0:
        print(f"{col}: {outlier_count} outliers ({outlier_count/df.shape[0]*100:.2f}%)")
        
        # For balance, we'll cap the outliers instead of removing them
        if col == 'balance':
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            print(f"  - Capped outliers to range: [{lower_bound:.2f}, {upper_bound:.2f}]")
        # For campaign, we'll cap the outliers
        elif col == 'campaign':
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            print(f"  - Capped outliers to range: [{lower_bound:.2f}, {upper_bound:.2f}]")
        # For duration, we'll keep as is since it's an important predictor
        elif col == 'duration':
            print(f"  - Kept outliers as is (important predictor)")
        # For pdays, special handling since -1 means client was not contacted
        elif col == 'pdays':
            # Keep -1 values, cap only positive outliers
            mask = (df[col] > 0) & (df[col] > upper_bound)
            df.loc[mask, col] = upper_bound
            print(f"  - Kept -1 values, capped positive outliers to: {upper_bound:.2f}")
        # For other columns, we'll keep outliers
        else:
            print(f"  - Kept outliers as is")

# 3. Convert categorical variables
print("\n=== Converting categorical variables ===")

# Binary categorical variables (yes/no)
binary_cols = ['default', 'housing', 'loan']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})
    print(f"Converted {col} to binary (1/0)")

# Convert target variable
df['deposit'] = df['deposit'].map({'yes': 1, 'no': 0})
print(f"Converted deposit to binary (1/0)")

# Create dummy variables for other categorical columns
# We'll exclude 'deposit' as it's our target
categorical_cols = [col for col in categorical_cols if col not in binary_cols + ['deposit']]
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print(f"Created dummy variables for {categorical_cols}")
print(f"New shape after encoding: {df_encoded.shape}")

# 4. Save the cleaned data
print("\n=== Saving cleaned data ===")
df_encoded.to_csv('./data/bank_cleaned.csv', index=False)
print("Cleaned data saved to './data/bank_cleaned.csv'")

# 5. Create visualizations of cleaned data
print("\n=== Creating visualizations of cleaned data ===")

# Distribution of numerical features after cleaning
for col in numerical_cols:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col} (After Cleaning)')
    plt.savefig(f'./visualizations/cleaned/{col}_distribution.png')
    plt.close()

# Correlation heatmap after cleaning
plt.figure(figsize=(12, 10))
correlation_matrix = df_encoded.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix (After Cleaning)')
plt.tight_layout()
plt.savefig('./visualizations/cleaned/correlation_heatmap.png')
plt.close()

# Compare original vs cleaned data for key numerical features
for col in ['balance', 'campaign', 'duration']:
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(x='deposit', y=col, data=df_original)
    plt.title(f'{col} by Target (Original)')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x='deposit', y=col, data=df)
    plt.title(f'{col} by Target (Cleaned)')
    
    plt.tight_layout()
    plt.savefig(f'./visualizations/cleaned/{col}_comparison.png')
    plt.close()

print("\nData cleaning completed. Visualizations saved in './visualizations/cleaned/' directory.")
