import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# Set the style for plots
sns.set_theme(style="whitegrid")
plt.style.use('seaborn-v0_8-whitegrid')

# Create directories for saving plots and models
os.makedirs('./visualizations/models', exist_ok=True)
os.makedirs('./models', exist_ok=True)

# Load the train and test data
print("Loading the train and test data...")
X_train = pd.read_csv('./data/X_train.csv')
y_train = pd.read_csv('./data/y_train.csv')
X_test = pd.read_csv('./data/X_test.csv')
y_test = pd.read_csv('./data/y_test.csv')

# Convert y_train and y_test to 1D arrays
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Define stratified k-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define a function to perform grid search with cross-validation
def tune_hyperparameters(model, param_grid, X_train, y_train, model_name):
    print(f"\n=== Tuning Hyperparameters for {model_name} ===")
    
    # Create grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    # Print best parameters and score
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

# Define a function to evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    # Train the model if it hasn't been trained yet
    if not hasattr(model, 'classes_'):
        model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Print metrics
    print(f"\n=== {model_name} Performance ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'./visualizations/models/{model_name.replace(" ", "_").lower()}_confusion_matrix.png')
    plt.close()
    
    # Create ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.savefig(f'./visualizations/models/{model_name.replace(" ", "_").lower()}_roc_curve.png')
    plt.close()
    
    # Create precision-recall curve
    plt.figure(figsize=(8, 6))
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.plot(recall_curve, precision_curve)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.savefig(f'./visualizations/models/{model_name.replace(" ", "_").lower()}_pr_curve.png')
    plt.close()
    
    # Save model
    joblib.dump(model, f'./models/{model_name.replace(" ", "_").lower()}.pkl')
    
    # Return metrics for comparison
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }

# 1. Tune and Train Logistic Regression
print("\n=== Training Logistic Regression with Hyperparameter Tuning ===")
lr_param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['liblinear', 'saga', 'newton-cg'],
    'max_iter': [10000]
}
lr_model = tune_hyperparameters(
    LogisticRegression(random_state=42),
    lr_param_grid,
    X_train,
    y_train,
    "Logistic Regression"
)
lr_metrics = evaluate_model(lr_model, X_train, X_test, y_train, y_test, "Logistic Regression")

# 2. Tune and Train Decision Tree
print("\n=== Training Decision Tree with Hyperparameter Tuning ===")
dt_param_grid = {
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
dt_model = tune_hyperparameters(
    DecisionTreeClassifier(random_state=42),
    dt_param_grid,
    X_train,
    y_train,
    "Decision Tree"
)
dt_metrics = evaluate_model(dt_model, X_train, X_test, y_train, y_test, "Decision Tree")

# 3. Tune and Train Random Forest
print("\n=== Training Random Forest with Hyperparameter Tuning ===")
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_model = tune_hyperparameters(
    RandomForestClassifier(random_state=42),
    rf_param_grid,
    X_train,
    y_train,
    "Random Forest"
)
rf_metrics = evaluate_model(rf_model, X_train, X_test, y_train, y_test, "Random Forest")

# 4. Tune and Train Gradient Boosting
print("\n=== Training Gradient Boosting with Hyperparameter Tuning ===")
gb_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}
gb_model = tune_hyperparameters(
    GradientBoostingClassifier(random_state=42),
    gb_param_grid,
    X_train,
    y_train,
    "Gradient Boosting"
)
gb_metrics = evaluate_model(gb_model, X_train, X_test, y_train, y_test, "Gradient Boosting")

# 5. Tune and Train XGBoost
print("\n=== Training XGBoost with Hyperparameter Tuning ===")
xgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0]
}
xgb_model = tune_hyperparameters(
    XGBClassifier(random_state=42),
    xgb_param_grid,
    X_train,
    y_train,
    "XGBoost"
)
xgb_metrics = evaluate_model(xgb_model, X_train, X_test, y_train, y_test, "XGBoost")


# 7. Tune and Train KNN
print("\n=== Training KNN with Hyperparameter Tuning ===")
knn_param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]  # 1 for manhattan_distance, 2 for euclidean_distance
}
knn_model = tune_hyperparameters(
    KNeighborsClassifier(),
    knn_param_grid,
    X_train,
    y_train,
    "KNN"
)
knn_metrics = evaluate_model(knn_model, X_train, X_test, y_train, y_test, "KNN")

# Collect all metrics for comparison
all_metrics = [lr_metrics, dt_metrics, rf_metrics, gb_metrics, xgb_metrics, knn_metrics]
metrics_df = pd.DataFrame(all_metrics)

# Save metrics to CSV
metrics_df.to_csv('./models/model_comparison.csv', index=False)

# Create comparison plots
plt.figure(figsize=(12, 8))
metrics_df.set_index('model_name')[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']].plot(kind='bar')
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('./visualizations/models/model_comparison.png')
plt.close()

# Find the best model based on ROC AUC
best_model_idx = metrics_df['roc_auc'].idxmax()
best_model_name = metrics_df.loc[best_model_idx, 'model_name']
best_model_metrics = metrics_df.loc[best_model_idx]

print("\n=== Best Model ===")
print(f"Best Model: {best_model_name}")
print(f"Accuracy: {best_model_metrics['accuracy']:.4f}")
print(f"Precision: {best_model_metrics['precision']:.4f}")
print(f"Recall: {best_model_metrics['recall']:.4f}")
print(f"F1 Score: {best_model_metrics['f1']:.4f}")
print(f"ROC AUC: {best_model_metrics['roc_auc']:.4f}")

# Save best model name for later use
with open('./models/best_model.txt', 'w') as f:
    f.write(best_model_name)

print("\nModel training completed. All models saved in './models/' directory.")
print("Model comparison visualizations saved in './visualizations/models/' directory.")
