import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

# Set the style for plots
sns.set(style="whitegrid")
plt.style.use('seaborn-v0_8-whitegrid')

# Create directories for saving plots
os.makedirs('./visualizations/advanced', exist_ok=True)

# Load the engineered data
print("Loading the engineered dataset...")
df = pd.read_csv('./data/bank_engineered.csv')

# Load train and test data
X_train = pd.read_csv('./data/X_train.csv')
y_train = pd.read_csv('./data/y_train.csv')
X_test = pd.read_csv('./data/X_test.csv')
y_test = pd.read_csv('./data/y_test.csv')

print(f"Dataset shape: {df.shape}")

# 1. Target Distribution Visualization
print("\n=== Creating Target Distribution Visualization ===")
plt.figure(figsize=(10, 6))
target_counts = df['deposit'].value_counts()
ax = sns.countplot(x='deposit', data=df, palette='viridis')

# Add percentage labels
total = len(df)
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='bottom')

plt.title('Target Distribution (Deposit)', fontsize=15)
plt.xlabel('Deposit (1=Yes, 0=No)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.savefig('./visualizations/advanced/target_distribution.png')
plt.close()

# 2. Feature Importance Visualization
print("\n=== Creating Feature Importance Visualization ===")
# Using SelectKBest to get feature importance scores
selector = SelectKBest(f_classif, k='all')
selector.fit(X_train, y_train.values.ravel())

# Get feature importance scores
feature_scores = pd.DataFrame({
    'Feature': X_train.columns,
    'Score': selector.scores_
})

# Sort by importance
feature_scores = feature_scores.sort_values('Score', ascending=False)

# Plot top 20 features
plt.figure(figsize=(12, 10))
sns.barplot(x='Score', y='Feature', data=feature_scores.head(20), palette='viridis')
plt.title('Top 20 Features by Importance (ANOVA F-value)', fontsize=15)
plt.xlabel('F-Score', fontsize=12)
plt.tight_layout()
plt.savefig('./visualizations/advanced/feature_importance.png')
plt.close()

# 3. Correlation Heatmap
print("\n=== Creating Correlation Heatmap ===")
# Select top 15 features by importance plus target
top_features = feature_scores.head(15)['Feature'].tolist()
top_features.append('deposit')

# Create correlation matrix
correlation_matrix = df[top_features].corr()

# Plot heatmap
plt.figure(figsize=(14, 12))
mask = np.triu(correlation_matrix)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
            linewidths=0.5, mask=mask)
plt.title('Correlation Heatmap of Top Features', fontsize=15)
plt.tight_layout()
plt.savefig('./visualizations/advanced/correlation_heatmap.png')
plt.close()

# 4. Age Distribution by Deposit Status
print("\n=== Creating Age Distribution Visualization ===")
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='age', hue='deposit', kde=True, bins=30, 
             element='step', palette='viridis')
plt.title('Age Distribution by Deposit Status', fontsize=15)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.savefig('./visualizations/advanced/age_distribution.png')
plt.close()

# 5. Balance Distribution by Deposit Status
print("\n=== Creating Balance Distribution Visualization ===")
plt.figure(figsize=(12, 6))
sns.boxplot(x='deposit', y='balance', data=df, palette='viridis')
plt.title('Balance Distribution by Deposit Status', fontsize=15)
plt.xlabel('Deposit (1=Yes, 0=No)', fontsize=12)
plt.ylabel('Balance', fontsize=12)
plt.savefig('./visualizations/advanced/balance_distribution.png')
plt.close()

# 6. Duration Distribution by Deposit Status
print("\n=== Creating Duration Distribution Visualization ===")
plt.figure(figsize=(12, 6))
sns.boxplot(x='deposit', y='duration', data=df, palette='viridis')
plt.title('Call Duration Distribution by Deposit Status', fontsize=15)
plt.xlabel('Deposit (1=Yes, 0=No)', fontsize=12)
plt.ylabel('Duration (seconds)', fontsize=12)
plt.savefig('./visualizations/advanced/duration_distribution.png')
plt.close()

# 7. Campaign Distribution by Deposit Status
print("\n=== Creating Campaign Distribution Visualization ===")
plt.figure(figsize=(12, 6))
sns.boxplot(x='deposit', y='campaign', data=df, palette='viridis')
plt.title('Number of Contacts Distribution by Deposit Status', fontsize=15)
plt.xlabel('Deposit (1=Yes, 0=No)', fontsize=12)
plt.ylabel('Number of Contacts', fontsize=12)
plt.savefig('./visualizations/advanced/campaign_distribution.png')
plt.close()

# 8. PCA Visualization
print("\n=== Creating PCA Visualization ===")
# Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)

# Create DataFrame for plotting
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['deposit'] = y_train.values

# Plot PCA results
plt.figure(figsize=(12, 8))
sns.scatterplot(x='PC1', y='PC2', hue='deposit', data=pca_df, 
                palette='viridis', alpha=0.7)
plt.title('PCA: First Two Principal Components', fontsize=15)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance explained)', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance explained)', fontsize=12)
plt.savefig('./visualizations/advanced/pca_visualization.png')
plt.close()

# 9. Job Type vs Deposit
print("\n=== Creating Job Type Visualization ===")
# Extract job columns
job_columns = [col for col in df.columns if col.startswith('job_')]

# Create a new dataframe with job types
job_df = pd.DataFrame()
for col in job_columns:
    job_name = col.replace('job_', '')
    job_df = pd.concat([job_df, df[df[col] == 1][['deposit']].assign(job=job_name)])

# Plot job distribution
plt.figure(figsize=(14, 8))
job_counts = job_df.groupby('job')['deposit'].agg(['count', 'mean'])
job_counts = job_counts.sort_values('mean', ascending=False)

# Plot job success rate
ax = sns.barplot(x=job_counts.index, y='mean', data=job_counts, palette='viridis')
plt.title('Deposit Success Rate by Job Type', fontsize=15)
plt.xlabel('Job Type', fontsize=12)
plt.ylabel('Success Rate', fontsize=12)
plt.xticks(rotation=45, ha='right')

# Add percentage labels
for p in ax.patches:
    percentage = f'{p.get_height():.1%}'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='bottom')

plt.tight_layout()
plt.savefig('./visualizations/advanced/job_success_rate.png')
plt.close()

# 10. Month vs Deposit
print("\n=== Creating Month Visualization ===")
# Extract month columns
month_columns = [col for col in df.columns if col.startswith('month_')]

# Create a new dataframe with months
month_df = pd.DataFrame()
for col in month_columns:
    month_name = col.replace('month_', '')
    month_df = pd.concat([month_df, df[df[col] == 1][['deposit']].assign(month=month_name)])

# Define month order
month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
month_df['month'] = pd.Categorical(month_df['month'], categories=month_order, ordered=True)

# Plot month success rate
plt.figure(figsize=(14, 8))
month_counts = month_df.groupby('month')['deposit'].agg(['count', 'mean'])
month_counts = month_counts.reset_index()
month_counts = month_counts.sort_values('month')

ax = sns.barplot(x='month', y='mean', data=month_counts, palette='viridis')
plt.title('Deposit Success Rate by Month', fontsize=15)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Success Rate', fontsize=12)

# Add percentage labels
for p in ax.patches:
    percentage = f'{p.get_height():.1%}'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='bottom')

plt.tight_layout()
plt.savefig('./visualizations/advanced/month_success_rate.png')
plt.close()

# 11. Education vs Deposit
print("\n=== Creating Education Visualization ===")
# Extract education columns
education_columns = [col for col in df.columns if col.startswith('education_')]

# Create a new dataframe with education levels
education_df = pd.DataFrame()
for col in education_columns:
    education_name = col.replace('education_', '')
    education_df = pd.concat([education_df, df[df[col] == 1][['deposit']].assign(education=education_name)])

# Plot education success rate
plt.figure(figsize=(12, 8))
education_counts = education_df.groupby('education')['deposit'].agg(['count', 'mean'])
education_counts = education_counts.sort_values('mean', ascending=False)

ax = sns.barplot(x=education_counts.index, y='mean', data=education_counts, palette='viridis')
plt.title('Deposit Success Rate by Education Level', fontsize=15)
plt.xlabel('Education Level', fontsize=12)
plt.ylabel('Success Rate', fontsize=12)

# Add percentage labels
for p in ax.patches:
    percentage = f'{p.get_height():.1%}'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='bottom')

plt.tight_layout()
plt.savefig('./visualizations/advanced/education_success_rate.png')
plt.close()

# 12. Feature Pair Relationships
print("\n=== Creating Feature Pair Relationships Visualization ===")
# Select important numerical features
numerical_features = ['age', 'balance', 'duration', 'campaign', 'previous']
numerical_features.append('deposit')

# Create pairplot
plt.figure(figsize=(15, 15))
sns.pairplot(df[numerical_features], hue='deposit', palette='viridis', 
             diag_kind='kde', plot_kws={'alpha': 0.6})
plt.suptitle('Relationships Between Key Numerical Features', fontsize=16, y=1.02)
plt.savefig('./visualizations/advanced/feature_pairplot.png')
plt.close()

print("\nVisualization creation completed. All visualizations saved in './visualizations/advanced/' directory.")
