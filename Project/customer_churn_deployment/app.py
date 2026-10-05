from flask import Flask, render_template, request
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

app = Flask(__name__)
MODEL_FILE = 'customer_churn_elastic_net.pkl'
DATA_FILE = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'

numerical_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [
    'gender','Partner','Dependents','PhoneService','MultipleLines',
    'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection',
    'TechSupport','StreamingTV','StreamingMovies','Contract',
    'PaperlessBilling','PaymentMethod'
]
features = numerical_features + categorical_features

DATA_URLS = [
    'https://raw.githubusercontent.com/Giskard-AI/examples/main/datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv',
    'https://raw.githubusercontent.com/aiplanethub/Datasets/master/WA_Fn-UseC_-Telco-Customer-Churn.csv'
]

def train_model():
    if not os.path.exists(DATA_FILE):
        last_error = None
        for url in DATA_URLS:
            try:
                df = pd.read_csv(url)
                df.to_csv(DATA_FILE, index=False)
                break
            except Exception as e:
                last_error = e
        else:
            raise RuntimeError(f'Could not download dataset: {last_error}')
    else:
        df = pd.read_csv(DATA_FILE)

    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    model_df = df[features + ['Churn']].copy()
    X = model_df[features]
    y = model_df['Churn'].map({'No': 0, 'Yes': 1})

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    model = LogisticRegression(
        penalty='elasticnet', solver='saga', l1_ratio=0.5,
        C=1.0, max_iter=5000, random_state=42
    )
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_FILE)
    return pipeline

def get_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return train_model()

model = get_model()

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    probability = None
    error = None
    form_data = request.form.to_dict() if request.method == 'POST' else {}

    if request.method == 'POST':
        try:
            data = {
                'SeniorCitizen': int(request.form['SeniorCitizen']),
                'tenure': float(request.form['tenure']),
                'MonthlyCharges': float(request.form['MonthlyCharges']),
                'TotalCharges': float(request.form['TotalCharges']),
                'gender': request.form['gender'],
                'Partner': request.form['Partner'],
                'Dependents': request.form['Dependents'],
                'PhoneService': request.form['PhoneService'],
                'MultipleLines': request.form['MultipleLines'],
                'InternetService': request.form['InternetService'],
                'OnlineSecurity': request.form['OnlineSecurity'],
                'OnlineBackup': request.form['OnlineBackup'],
                'DeviceProtection': request.form['DeviceProtection'],
                'TechSupport': request.form['TechSupport'],
                'StreamingTV': request.form['StreamingTV'],
                'StreamingMovies': request.form['StreamingMovies'],
                'Contract': request.form['Contract'],
                'PaperlessBilling': request.form['PaperlessBilling'],
                'PaymentMethod': request.form['PaymentMethod']
            }
            input_df = pd.DataFrame([data], columns=features)
            prediction = int(model.predict(input_df)[0])
            probability = float(model.predict_proba(input_df)[0][1]) * 100
            result = 'Customer Likely to Churn' if prediction == 1 else 'Customer Not Likely to Churn'
        except Exception as e:
            error = str(e)

    return render_template('index.html', result=result, probability=probability, error=error, form_data=form_data)

@app.route('/health')
def health():
    return {'status': 'ok', 'model': 'Elastic Net Logistic Regression'}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
