import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# Set the style for plots
sns.set(style="whitegrid")
plt.style.use('seaborn-v0_8-whitegrid')

# Create directories for saving plots
os.makedirs('./visualizations/feature_engineering', exist_ok=True)

# Load the cleaned data
print("Loading the cleaned dataset...")
df = pd.read_csv('./data/bank_cleaned.csv')

print(f"Original shape after cleaning: {df.shape}")

# 1. Create new features
print("\n=== Creating new features ===")

# Age groups
print("Creating age groups...")
df['age_group'] = pd.cut(df['age'], bins=[17, 30, 40, 50, 60, 100], 
                         labels=['18-30', '31-40', '41-50', '51-60', '60+'])
df = pd.get_dummies(df, columns=['age_group'], prefix='age_group')

# Balance categories
print("Creating balance categories...")
df['balance_category'] = pd.cut(df['balance'], bins=[-10000, 0, 1000, 5000, 100000], 
                               labels=['negative', 'low', 'medium', 'high'])
df = pd.get_dummies(df, columns=['balance_category'], prefix='balance')

# Campaign categories
print("Creating campaign categories...")
df['campaign_category'] = pd.cut(df['campaign'], bins=[-1, 1, 3, 10], 
                                labels=['single', 'few', 'many'])
df = pd.get_dummies(df, columns=['campaign_category'], prefix='campaign')

# Previous contact binary
print("Creating previous contact binary feature...")
df['previously_contacted'] = (df['previous'] > 0).astype(int)

# Duration categories
print("Creating duration categories...")
df['call_duration_category'] = pd.cut(df['duration'], bins=[0, 180, 600, 3000], 
                                     labels=['short', 'medium', 'long'])
df = pd.get_dummies(df, columns=['call_duration_category'], prefix='duration')

# 2. Feature interactions
print("\n=== Creating feature interactions ===")

# Interaction between age and job
print("Creating age-job interaction features...")
# We'll use the job dummy variables that already exist
job_columns = [col for col in df.columns if col.startswith('job_')]
for job_col in job_columns:
    df[f'age_{job_col}'] = df['age'] * df[job_col]

# Interaction between balance and housing
print("Creating balance-housing interaction...")
df['balance_housing'] = df['balance'] * df['housing']

# Interaction between duration and previous
print("Creating duration-previous interaction...")
df['duration_previous'] = df['duration'] * df['previously_contacted']

# 3. Feature scaling
print("\n=== Scaling numerical features ===")
numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

# Create scaled versions of numerical features
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])

print(f"Final shape after feature engineering: {df_scaled.shape}")

# 4. Split the data into train and test sets
print("\n=== Splitting data into train and test sets ===")
X = df_scaled.drop('deposit', axis=1)
y = df_scaled['deposit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# 5. Save the engineered data
print("\n=== Saving engineered data ===")
# Save the full engineered dataset
df_scaled.to_csv('./data/bank_engineered.csv', index=False)
print("Full engineered data saved to './data/bank_engineered.csv'")

# Save the train and test sets
X_train.to_csv('./data/X_train.csv', index=False)
y_train.to_csv('./data/y_train.csv', index=False)
X_test.to_csv('./data/X_test.csv', index=False)
y_test.to_csv('./data/y_test.csv', index=False)
print("Train and test sets saved to data directory")

# 6. Create visualizations of engineered features
print("\n=== Creating visualizations of engineered features ===")

# Feature importance based on correlation with target
plt.figure(figsize=(12, 10))
correlations = df_scaled.corr()['deposit'].sort_values(ascending=False)
top_correlations = correlations[1:21]  # Top 20 features excluding target itself
top_correlations.plot(kind='bar')
plt.title('Top 20 Features by Correlation with Target')
plt.tight_layout()
plt.savefig('./visualizations/feature_engineering/feature_correlations.png')
plt.close()

# Distribution of key engineered features
engineered_features = ['previously_contacted', 'balance_housing', 'duration_previous']
for feature in engineered_features:
    plt.figure(figsize=(10, 6))
    if df_scaled[feature].nunique() <= 2:
        # For binary features
        sns.countplot(x=feature, hue='deposit', data=df_scaled)
    else:
        # For continuous features
        sns.histplot(df_scaled[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.savefig(f'./visualizations/feature_engineering/{feature}_distribution.png')
    plt.close()

# Pairplot of selected features
selected_features = ['age', 'balance', 'duration', 'previously_contacted', 'deposit']
plt.figure(figsize=(12, 10))
sns.pairplot(df_scaled[selected_features], hue='deposit')
plt.savefig('./visualizations/feature_engineering/pairplot.png')
plt.close()

print("\nFeature engineering completed. Visualizations saved in './visualizations/feature_engineering/' directory.")

# Save the scaler for later use in the Streamlit app
import joblib
os.makedirs('F:\\home\\ubuntu\\bank_marketing_project\\models', exist_ok=True)
joblib.dump(scaler, './models/scaler.pkl')
print("Scaler saved to './models/scaler.pkl' for use in the Streamlit app")
