import os
import sys
import zipfile

# Create directories for saving files
os.makedirs('./deployment', exist_ok=True)

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

# Create a zip file for server deployment
deployment_files = [
    './streamlit_app.py',
    './requirements.txt',
    './README.md',
    './data/bank_engineered.csv'
]

try:
    zip_path = './deployment/bank_marketing_app.zip'
    if os.path.exists(zip_path):
        os.remove(zip_path)
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add individual files
        for file in deployment_files:
            if os.path.exists(file):
                zipf.write(file, os.path.basename(file))
        
        # Add models directory
        if os.path.exists('./models'):
            for root, dirs, files in os.walk('./models'):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.join('models', os.path.basename(file_path))
                    zipf.write(file_path, arcname)
    
    print("\nServer deployment package created successfully!")
    print("Contents:")
    print("1. streamlit_app.py - Main application")
    print("2. requirements.txt - Dependencies")
    print("3. models/ - Trained models")
    print("4. data/bank_engineered.csv - Dataset")
    print("\nTo deploy on server:")
    print("1. Upload bank_marketing_app.zip to server")
    print("2. Unzip the package")
    print("3. Run: pip install -r requirements.txt")
    print("4. Start app: streamlit run streamlit_app.py")

except Exception as e:
    print(f"\nError creating deployment package: {str(e)}")
