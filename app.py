import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
import multiprocessing

# Suppress warnings and set CPU cores
warnings.filterwarnings('ignore')
os.environ["LOKY_MAX_CPU_COUNT"] = str(multiprocessing.cpu_count())
os.environ["JOBLIB_CPU_COUNT"] = "1"  # Force single CPU to avoid Windows subprocess issues

# Set page configuration
st.set_page_config(
    page_title="Bank Marketing Campaign Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .card {
        border-radius: 5px;
        background-color: #f9f9f9;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    .highlight {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        background-color: #f5f5f5;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #f0f8ff;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .model-selector {
        padding: 10px;
        background-color: #e6f2ff;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .tab-content {
        padding: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load the best model
@st.cache_resource
def load_model():
    try:
        with open('./models/best_model.txt', 'r') as f:
            best_model_name = f.read().strip()
        model_path = f'./models/{best_model_name.replace(" ", "_").lower()}.pkl'
        model = joblib.load(model_path)
        return model, best_model_name
    except:
        # If best model file doesn't exist, try to load any model
        model_files = [f for f in os.listdir('./models') if f.endswith('.pkl') and f != 'scaler.pkl']
        if model_files:
            model = joblib.load(f'./models/{model_files[0]}')
            return model, model_files[0].replace('.pkl', '').replace('_', ' ').title()
        else:
            return None, "No model found"

# Load all available models
@st.cache_resource
def load_all_models():
    models = {}
    try:
        # Load all saved model files except scaler.pkl
        model_files = [f for f in os.listdir('./models') 
                      if f.endswith('.pkl') and f != 'scaler.pkl']
        
        for model_file in model_files:
            model_name = model_file.replace('.pkl', '').replace('_', ' ').title()
            model = joblib.load(f'./models/{model_file}')
            models[model_name] = model
            
        print(f"Loaded {len(models)} models: {list(models.keys())}")
        return models
            
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return {}

# Load the scaler
@st.cache_resource
def load_scaler():
    try:
        scaler = joblib.load('./models/scaler.pkl')
        return scaler
    except:
        return None

# Load the data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('./data/bank_engineered.csv')
        return df
    except:
        try:
            df = pd.read_csv('./data/bank_cleaned.csv')
            return df
        except:
            try:
                df = pd.read_csv('./data/bank.csv')
                return df
            except:
                return None

# Load the original data for visualizations
@st.cache_data
def load_original_data():
    try:
        df = pd.read_csv('./data/bank.csv')
        return df
    except:
        
        return print('No data found!')

# Function to preprocess input data
def preprocess_input(input_data, df):
    # Create a DataFrame with all features set to 0
    features = pd.DataFrame(0, index=[0], columns=df.drop('deposit', axis=1).columns)
    
    # Set numerical features
    numerical_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    for feature in numerical_features:
        features[feature] = input_data[feature]
    
    # Set binary features
    if input_data['default'] == 'Yes':
        features['default'] = 1
    if input_data['housing'] == 'Yes':
        features['housing'] = 1
    if input_data['loan'] == 'Yes':
        features['loan'] = 1
    
    # Set categorical features
    # Job
    job_col = f"job_{input_data['job'].lower()}"
    if job_col in features.columns:
        features[job_col] = 1
    
    # Marital
    marital_col = f"marital_{input_data['marital'].lower()}"
    if marital_col in features.columns:
        features[marital_col] = 1
    
    # Education
    education_col = f"education_{input_data['education'].lower()}"
    if education_col in features.columns:
        features[education_col] = 1
    
    # Contact
    contact_col = f"contact_{input_data['contact'].lower()}"
    if contact_col in features.columns:
        features[contact_col] = 1
    
    # Month
    month_col = f"month_{input_data['month'].lower()}"
    if month_col in features.columns:
        features[month_col] = 1
    
    # Poutcome
    poutcome_col = f"poutcome_{input_data['poutcome'].lower()}"
    if poutcome_col in features.columns:
        features[poutcome_col] = 1
    
    # Create derived features
    # Age groups
    age = input_data['age']
    if 18 <= age <= 30:
        features['age_group_18-30'] = 1
    elif 31 <= age <= 40:
        features['age_group_31-40'] = 1
    elif 41 <= age <= 50:
        features['age_group_41-50'] = 1
    elif 51 <= age <= 60:
        features['age_group_51-60'] = 1
    else:
        features['age_group_60+'] = 1
    
    # Balance categories
    balance = input_data['balance']
    if balance < 0:
        features['balance_negative'] = 1
    elif 0 <= balance < 1000:
        features['balance_low'] = 1
    elif 1000 <= balance < 5000:
        features['balance_medium'] = 1
    else:
        features['balance_high'] = 1
    
    # Campaign categories
    campaign = input_data['campaign']
    if campaign <= 1:
        features['campaign_single'] = 1
    elif 2 <= campaign <= 3:
        features['campaign_few'] = 1
    else:
        features['campaign_many'] = 1
    
    # Previous contact binary
    features['previously_contacted'] = 1 if input_data['previous'] > 0 else 0
    
    # Duration categories
    duration = input_data['duration']
    if duration <= 180:
        features['duration_short'] = 1
    elif 181 <= duration <= 600:
        features['duration_medium'] = 1
    else:
        features['duration_long'] = 1
    
    # Feature interactions
    # Age-job interaction
    job_col = f"job_{input_data['job'].lower()}"
    if job_col in features.columns:
        features[f"age_{job_col}"] = input_data['age'] * features[job_col]
    
    # Balance-housing interaction
    features['balance_housing'] = input_data['balance'] * features['housing']
    
    # Duration-previous interaction
    features['duration_previous'] = input_data['duration'] * features['previously_contacted']
    
    return features

# Function to make prediction
def predict(model, scaler, features):
    # Scale numerical features
    numerical_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    if scaler is not None:
        features[numerical_features] = scaler.transform(features[numerical_features])
    
    # Make prediction
    prediction_proba = model.predict_proba(features)[0, 1]
    prediction = 1 if prediction_proba >= 0.5 else 0
    
    return prediction, prediction_proba

# Function to display feature importance
def display_feature_importance(model, features):
    if hasattr(model, 'feature_importances_'):
        # For tree-based models (Random Forest, XGBoost, etc.)
        importances = model.feature_importances_
        indices = np.argsort(importances)[-15:]  # Top 15 features
        
        fig = px.bar(
            x=importances[indices],
            y=[features.columns[i] for i in indices],
            orientation='h',
            title='Top 15 Most Important Features',
            labels={'x': 'Feature Importance', 'y': 'Feature'},
            color=importances[indices],
            color_continuous_scale='Blues'
        )
        
    elif hasattr(model, 'coef_'):
        # For linear models (Logistic Regression)
        coefficients = np.abs(model.coef_[0])  # Using absolute values for importance
        indices = np.argsort(coefficients)[-15:]  # Top 15 features
        
        fig = px.bar(
            x=coefficients[indices],
            y=[features.columns[i] for i in indices],
            orientation='h',
            title='Top 15 Most Influential Features (Absolute Coefficient Values)',
            labels={'x': 'Absolute Coefficient Value', 'y': 'Feature'},
            color=coefficients[indices],
            color_continuous_scale='Blues'
        )
        
    fig.update_layout(
        height=500,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Add explanation based on model type
    if hasattr(model, 'coef_'):
        st.markdown("""
        > **Note**: For Logistic Regression, feature importance is shown using absolute coefficient values. 
        Larger values indicate stronger influence on the prediction, regardless of whether the influence 
        is positive or negative.
        """)
    else:
        st.markdown("""
        > **Note**: Feature importance shows the relative contribution of each feature to the model's predictions. 
        Higher values indicate more important features.
        """)

# Function to prepare data for model evaluation
def prepare_evaluation_data(df):
    try:
        # Load the preprocessed test data
        X_test = pd.read_csv('./data/X_test.csv')
        y_test = pd.read_csv('./data/y_test.csv')
        
        # Convert y_test to 1D array if needed
        if isinstance(y_test, pd.DataFrame):
            y_test = y_test.values.ravel()
            
        return None, X_test, None, y_test
        
    except Exception as e:
        st.error(f"Error loading test data: {str(e)}")
        return None, None, None, None

# Function to evaluate model performance
def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Calculate precision-recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall_curve, precision_curve)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'roc': (fpr, tpr, roc_auc),
        'pr': (recall_curve, precision_curve, pr_auc)
    }

# Function to display confusion matrix
def plot_confusion_matrix(cm):
    # Create a heatmap using Plotly
    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual"),
        x=['No Subscription', 'Subscription'],
        y=['No Subscription', 'Subscription'],
        color_continuous_scale='Blues'
    )
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        width=500,
        height=500
    )
    return fig

# Function to plot ROC curve
def plot_roc_curve(fpr, tpr, roc_auc):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr, 
            y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            line=dict(color='royalblue', width=2)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], 
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='gray', width=2, dash='dash')
        )
    )
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=600,
        height=500,
        showlegend=True
    )
    return fig

# Function to plot precision-recall curve
def plot_pr_curve(recall, precision, pr_auc):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recall, 
            y=precision,
            mode='lines',
            name=f'PR Curve (AUC = {pr_auc:.3f})',
            line=dict(color='forestgreen', width=2)
        )
    )
    fig.update_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        width=600,
        height=500,
        showlegend=True
    )
    return fig

# Function to display model comparison
def plot_model_comparison(models, X_test, y_test):
    try:
        # Load pre-calculated basic metrics
        df_comparison = pd.read_csv('./models/model_comparison.csv')
        
        # Load test data
        _, X_test, _, y_test = prepare_evaluation_data(None)
        
        # Create radar chart from pre-calculated metrics
        fig = go.Figure()
        
        results = {}
        for model_name in df_comparison['model_name']:
            display_name = model_name.replace('_', ' ').title()
            model_metrics = df_comparison[df_comparison['model_name'] == model_name].iloc[0]
            
            # Add trace to radar chart
            fig.add_trace(go.Scatterpolar(
                r=[model_metrics['accuracy'], 
                   model_metrics['precision'], 
                   model_metrics['recall'], 
                   model_metrics['f1'],
                   model_metrics['roc_auc']],
                theta=['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC'],
                name=display_name,
                fill='toself',
                line=dict(width=2)
            ))
            
            # Calculate additional metrics for each model
            if X_test is not None and y_test is not None and display_name in models:
                model = models[display_name]
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                
                # Calculate confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                
                # Store all metrics
                results[display_name] = {
                    'accuracy': model_metrics['accuracy'],
                    'precision': model_metrics['precision'],
                    'recall': model_metrics['recall'],
                    'f1': model_metrics['f1'],
                    'roc_auc': model_metrics['roc_auc'],
                    'confusion_matrix': cm,
                    'roc': (fpr, tpr, model_metrics['roc_auc'])
                }
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title='Model Performance Comparison',
            showlegend=True,
            height=600
        )
        
        return fig, results
        
    except Exception as e:
        st.error(f"Error in model comparison: {str(e)}")
        return None, {}

def main():
    # Load model, scaler, and data
    model, model_name = load_model()
    scaler = load_scaler()
    df = load_data()
    original_df = load_original_data()
    
    # Sidebar
    st.sidebar.title("Bank Marketing Campaign Predictor")
    st.sidebar.image("https://img.icons8.com/color/96/000000/bank-building.png", width=100)
    
    # Navigation
    page = st.sidebar.radio("Navigation", ["Predictor", "Data Insights", "Model Performance"])
    
    if page == "Predictor":
        st.markdown("<h1 class='main-header'>Bank Term Deposit Predictor</h1>", unsafe_allow_html=True)
        st.markdown("""
        <div class="highlight">
        This application predicts whether a client will subscribe to a term deposit based on various features.
        Fill in the client information below to get a prediction.
        </div>
        """, unsafe_allow_html=True)
        
        if model is None or df is None:
            st.error("Error: Model or data not found. Please make sure the model and data files are available.")
            return
        
        # Create input form
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<h3>Personal Information</h3>", unsafe_allow_html=True)
            age = st.number_input("Age", min_value=18, max_value=95, value=41)
            job_options = ['Admin.', 'Blue-collar', 'Entrepreneur', 'Housemaid', 'Management', 
                          'Retired', 'Self-employed', 'Services', 'Student', 'Technician', 'Unemployed', 'Unknown']
            job = st.selectbox("Job", job_options)
            marital_options = ['Single', 'Married', 'Divorced']
            marital = st.selectbox("Marital Status", marital_options)
            education_options = ['Primary', 'Secondary', 'Tertiary', 'Unknown']
            education = st.selectbox("Education", education_options)
        
        with col2:
            st.markdown("<h3>Financial Information</h3>", unsafe_allow_html=True)
            default = st.selectbox("Has Credit in Default?", ["No", "Yes"])
            balance = st.number_input("Account Balance (€)", min_value=-10000, max_value=100000, value=1500)
            housing = st.selectbox("Has Housing Loan?", ["No", "Yes"])
            loan = st.selectbox("Has Personal Loan?", ["No", "Yes"])
        
        with col3:
            st.markdown("<h3>Campaign Information</h3>", unsafe_allow_html=True)
            contact_options = ['Cellular', 'Telephone', 'Unknown']
            contact = st.selectbox("Contact Communication Type", contact_options)
            day = st.number_input("Last Contact Day of Month", min_value=1, max_value=31, value=15)
            month_options = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            month = st.selectbox("Last Contact Month", month_options)
            duration = st.number_input("Last Contact Duration (seconds)", min_value=0, max_value=5000, value=300)
            campaign = st.number_input("Number of Contacts During Campaign", min_value=1, max_value=50, value=2)
            pdays = st.number_input("Days Since Last Contact (-1 if no previous contact)", min_value=-1, max_value=999, value=-1)
            previous = st.number_input("Number of Previous Contacts", min_value=0, max_value=50, value=0)
            poutcome_options = ['Success', 'Failure', 'Unknown']
            poutcome = st.selectbox("Outcome of Previous Campaign", poutcome_options)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Create a dictionary with input values
        input_data = {
            'age': age,
            'job': job,
            'marital': marital,
            'education': education,
            'default': default,
            'balance': balance,
            'housing': housing,
            'loan': loan,
            'contact': contact,
            'day': day,
            'month': month,
            'duration': duration,
            'campaign': campaign,
            'pdays': pdays,
            'previous': previous,
            'poutcome': poutcome
        }
        
        # Make prediction when button is clicked
        if st.button("Predict Deposit Subscription"):
            # Preprocess input data
            features = preprocess_input(input_data, df)
            
            # Make prediction
            prediction, prediction_proba = predict(model, scaler, features)
            
            # Display prediction
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<h3>Prediction Result</h3>", unsafe_allow_html=True)
                if prediction == 1:
                    st.success("This client is **likely to subscribe** to a term deposit.")
                else:
                    st.error("This client is **unlikely to subscribe** to a term deposit.")
                
                st.markdown(f"<h4>Probability: {prediction_proba:.2%}</h4>", unsafe_allow_html=True)
                
                # Create a gauge chart for probability
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prediction_proba * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Subscription Probability"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 100], 'color': "lightblue"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("<h3>Key Factors</h3>", unsafe_allow_html=True)
                
                # Display key factors
                st.markdown("<h4>Top Factors Influencing Prediction:</h4>", unsafe_allow_html=True)
                
                # Duration is typically the most important feature
                if duration > 600:
                    st.markdown("- **Long call duration** (> 10 minutes) strongly indicates interest")
                elif duration > 180:
                    st.markdown("- **Medium call duration** (3-10 minutes) shows potential interest")
                else:
                    st.markdown("- **Short call duration** (< 3 minutes) often indicates disinterest")
                
                # Previous campaign success
                if poutcome == 'Success':
                    st.markdown("- **Previous campaign success** is a strong positive indicator")
                
                # Age factor
                if 25 <= age <= 35:
                    st.markdown("- **Young professional age group** (25-35) tends to be more receptive")
                elif age > 60:
                    st.markdown("- **Retirement age** clients often show interest in deposits")
                
                # Balance factor
                if balance > 5000:
                    st.markdown("- **High account balance** suggests financial capacity for deposits")
                
                # Display feature importance
                st.markdown("<h4>Feature Importance:</h4>", unsafe_allow_html=True)
                display_feature_importance(model, features)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Recommendations
            st.markdown("<h3 class='sub-header'>Recommendations</h3>", unsafe_allow_html=True)
            st.markdown("<div class='highlight'>", unsafe_allow_html=True)
            
            if prediction == 1:
                st.markdown("""
                ### For Likely Subscribers:
                1. **Immediate Follow-up**:
                   - Schedule a follow-up call within 48 hours
                   - Present detailed term deposit benefits
                   - Prepare personalized offer based on balance and profile
                
                2. **Focus Areas**:
                   - Highlight competitive interest rates
                   - Discuss flexible term lengths
                   - Emphasize account security features
                
                3. **Documentation**:
                   - Prepare necessary paperwork
                   - Send digital information package
                   - Set up online banking access
                """)
            else:
                st.markdown("""
                ### For Unlikely Subscribers:
                1. **Alternative Approach**:
                   - Consider revisiting after 3-6 months
                   - Explore other banking products that may better suit needs
                   - Document concerns for future reference
                
                2. **Improvement Areas**:
                   - Note client's specific objections
                   - Analyze timing and market conditions
                   - Consider different communication channels
                
                3. **Future Strategy**:
                   - Set reminder for future follow-up
                   - Prepare alternative product offerings
                   - Document preferred contact time/method
                """)
            
            st.markdown("</div>", unsafe_allow_html=True)

        
    elif page == "Data Insights":
        st.markdown("<h1 class='main-header'>Data Insights</h1>", unsafe_allow_html=True)
        
        if original_df is not None:
            tab1, tab2, tab3 = st.tabs(["Distribution Analysis", "Feature Relationships", "Target Analysis"])
            
            with tab1:
                st.subheader("Numerical Features Distribution")
                numerical_features = original_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                num_feature = st.selectbox(
                    "Select Numerical Feature",
                    numerical_features
                )
                try:
                    fig = px.histogram(original_df, x=num_feature, 
                                    marginal="box", 
                                    color="deposit",
                                    title=f"{num_feature.capitalize()} Distribution by Deposit Status")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating histogram: {str(e)}")
            
            with tab2:
                st.subheader("Feature Relationships")
                # Get unique column names
                available_features = original_df.columns.unique().tolist()
                
                col1, col2 = st.columns(2)
                with col1:
                    x_feat = st.selectbox("Select X-axis feature", available_features, key="x_feature")
                with col2:
                    # Remove selected x_feature from y_feature options to prevent duplicates
                    y_features = [f for f in available_features if f != x_feat]
                    y_feat = st.selectbox("Select Y-axis feature", y_features, key="y_feature")
                
                try:
                    if pd.api.types.is_numeric_dtype(original_df[x_feat]) and \
                    pd.api.types.is_numeric_dtype(original_df[y_feat]):
                        fig = px.scatter(
                            original_df,
                            x=x_feat,
                            y=y_feat,
                            color='deposit',
                            title=f"{x_feat} vs {y_feat}",
                            labels={
                                x_feat: x_feat.replace('_', ' ').title(),
                                y_feat: y_feat.replace('_', ' ').title()
                            }
                        )
                    else:
                        fig = px.box(
                            original_df,
                            x=x_feat,
                            y=y_feat,
                            color='deposit',
                            title=f"{x_feat} vs {y_feat}",
                            labels={
                                x_feat: x_feat.replace('_', ' ').title(),
                                y_feat: y_feat.replace('_', ' ').title()
                            }
                        )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating plot: {str(e)}")
            
            with tab3:
                st.subheader("Target Variable Analysis")
                try:
                    # Target distribution
                    target_dist = original_df['deposit'].value_counts(normalize=True)
                    fig = px.pie(
                        values=target_dist.values,
                        names=target_dist.index,
                        title="Deposit Subscription Distribution",
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Additional target analysis
                    st.subheader("Target Distribution by Key Features")
                    categorical_features = original_df.select_dtypes(include=['object']).columns.tolist()
                    categorical_features.remove('deposit')  # Remove target variable
                    
                    selected_cat_feature = st.selectbox(
                        "Select Categorical Feature",
                        categorical_features,
                        key="target_analysis"
                    )
                    
                    fig = px.bar(
                        original_df.groupby(selected_cat_feature)['deposit']
                        .value_counts(normalize=True)
                        .unstack(),
                        title=f"Deposit Distribution by {selected_cat_feature}",
                        barmode='group'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error in target analysis: {str(e)}")
        else:
            st.error("No data available for analysis. Please check if the data file is loaded correctly.")
        
    elif page == "Model Performance":
        st.markdown("<h1 class='main-header'>Model Performance</h1>", unsafe_allow_html=True)
        
        models = load_all_models()
        if models:
            # Load model comparison visualization and metrics
            fig_comparison, results = plot_model_comparison(models, None, None)
            
            if fig_comparison is not None:
                # Find best model based on ROC AUC
                best_model = max(results.items(), key=lambda x: x[1]['roc_auc'])
                best_model_name = best_model[0]
                best_metrics = best_model[1]
                
                # Display best model info
                st.markdown(
                    f"""
                    <div class="highlight">
                        <h3>Best Performing Model</h3>
                        <p>The <strong>{best_model_name}</strong> achieved the highest performance with:</p>
                        <ul>
                            <li>ROC AUC Score: {best_metrics['roc_auc']:.3f}</li>
                            <li>Accuracy: {best_metrics['accuracy']:.3f}</li>
                            <li>F1 Score: {best_metrics['f1']:.3f}</li>
                        </ul>
                        <p>This model is currently being used for making predictions in the Predictor tab.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Show model comparison radar chart
                st.subheader("Model Performance Comparison")
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Individual model analysis
                selected_model_name = st.selectbox("Select Model for Detailed Analysis", list(models.keys()))
                selected_model = models[selected_model_name]
                metrics = results[selected_model_name]
                
                # Display metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                with col2:
                    st.metric("Precision", f"{metrics['precision']:.3f}")
                with col3:
                    st.metric("Recall", f"{metrics['recall']:.3f}")
                with col4:
                    st.metric("F1 Score", f"{metrics['f1']:.3f}")
                
                # Display confusion matrix and ROC curve
                col1, col2 = st.columns(2)
                
                with col1:
                    # st.subheader("Confusion Matrix")
                    cm_fig = plot_confusion_matrix(metrics['confusion_matrix'])
                    st.plotly_chart(cm_fig, use_container_width=True)
                
                with col2:
                    # st.subheader("ROC Curve")
                    fpr, tpr, roc_auc = metrics['roc']
                    roc_fig = plot_roc_curve(fpr, tpr, roc_auc)
                    st.plotly_chart(roc_fig, use_container_width=True)
                
                # Load test data for feature importance
                _, X_test, _, _ = prepare_evaluation_data(df)
                if X_test is not None:
                    # Feature importance
                    st.subheader("Feature Importance")
                    display_feature_importance(selected_model, X_test)
            
        else:
            st.error("No models available for evaluation.")

if __name__ == "__main__":
    main()