import os
import sys

# Add project directory to path
sys.path.append('/home/ubuntu/bank_marketing_project')

# Create directories for saving files
os.makedirs('./deployment', exist_ok=True)

# Create README file
readme_content = """# Bank Marketing Campaign Project

## Overview
This project analyzes a bank marketing campaign dataset to predict whether clients will subscribe to a term deposit. The project includes:

1. **Data Exploration and Cleaning**: Analyzing the dataset structure, handling missing values and outliers
2. **Feature Engineering**: Creating new features and transforming existing ones
3. **Visualization**: Generating insights through comprehensive visualizations
4. **Model Training**: Implementing multiple machine learning models with hyperparameter tuning
5. **Streamlit App**: Interactive web application for predictions and insights

## Project Structure
- `data/`: Contains raw and processed datasets
- `models/`: Saved machine learning models
- `visualizations/`: Generated visualizations
- `app/`: Streamlit application files
- `*.py`: Python scripts for data processing and model training

## How to Run the Streamlit App
```bash
cd bank_marketing_project
streamlit run streamlit_app.py
```

## Features
- **Prediction**: Input client information to predict term deposit subscription
- **Data Insights**: Visualizations of key factors influencing subscription decisions
- **Model Performance**: Metrics and visualizations of model performance

## Models Implemented
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- SVM
- KNN

## Dataset
The dataset contains information about bank clients, marketing campaign details, and whether clients subscribed to a term deposit.

## Requirements
- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- streamlit
- plotly
- xgboost
"""

with open('./README.md', 'w') as f:
    f.write(readme_content)

# Create requirements.txt
requirements_content = """pandas==2.2.3
numpy==2.2.3
scikit-learn==1.6.1
matplotlib==3.10.1
seaborn==0.13.2
streamlit==1.32.0
plotly==5.18.0
xgboost==3.0.0
joblib==1.4.2
"""

with open('./requirements.txt', 'w') as f:
    f.write(requirements_content)

# Create deployment script
deployment_script = """#!/bin/bash

# Install required packages
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app.py
"""

with open('./run_app.sh', 'w') as f:
    f.write(deployment_script)

# Make the script executable
os.system('chmod +x ./run_app.sh')

# Create a zip file for deployment
import shutil

# Define files to include in the zip
files_to_zip = [
    './streamlit_app.py',
    './requirements.txt',
    './README.md',
    './run_app.sh',
    './data/bank_engineered.csv',
    './models'
]

# Check if models directory exists, if not create a placeholder
if not os.path.exists('./models'):
    os.makedirs('./models', exist_ok=True)
    with open('./models/README.txt', 'w') as f:
        f.write("Model files will be saved here after training completes.")
    files_to_zip.append('./models/README.txt')

# Create a zip file
shutil.make_archive('./deployment/bank_marketing_app', 'zip', '.', '.')




print("Deployment files created successfully!")
print("1. README.md - Project documentation")
