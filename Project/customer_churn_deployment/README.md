# Customer Churn Prediction - Elastic Net Logistic Regression

Deployment-ready Flask application for the Customer Churn Prediction project.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## Deploy on Render

1. Upload all files/folders in this ZIP to a GitHub repository.
2. In Render, create a new Web Service and connect the GitHub repository.
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `gunicorn app:app`
5. Deploy.

The app loads `customer_churn_elastic_net.pkl` if it exists. If it does not exist, it automatically downloads the standard Telco Customer Churn dataset, applies the same cleaning and 19-feature preprocessing, trains Elastic Net Logistic Regression, saves the model, and starts the prediction service.

## Model

- Logistic Regression with Elastic Net penalty
- Solver: SAGA
- L1 ratio: 0.5
- C: 1.0
- Max iterations: 5000
- StandardScaler for numeric features
- OneHotEncoder(handle_unknown='ignore') for categorical features
