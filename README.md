# Bank Marketing Campaign Project

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
streamlit run app.py
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
