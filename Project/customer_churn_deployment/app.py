
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

MODEL_FILE = "customer_churn_elastic_net.pkl"
DATA_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
HISTORY_FILE = "prediction_history.csv"

numerical_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_features = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

features = numerical_features + categorical_features

DATA_URLS = [
    "https://raw.githubusercontent.com/Giskard-AI/examples/main/datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/aiplanethub/Datasets/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
]


# ============================================================
# TRAIN THE MACHINE LEARNING MODEL
# ============================================================

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
            raise RuntimeError(
                f"Could not download dataset: {last_error}"
            )
    else:
        df = pd.read_csv(DATA_FILE)

    df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )
    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

    model_df = df[features + ["Churn"]].copy()
    X = model_df[features]
    y = model_df["Churn"].map({"No": 0, "Yes": 1})

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ])

    model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        C=1.0,
        max_iter=5000,
        random_state=42
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_FILE)

    return pipeline


# ============================================================
# LOAD MODEL
# ============================================================

def get_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)

    return train_model()


model = get_model()


# ============================================================
# CLASSIFY LOW, MEDIUM AND HIGH RISK
# ============================================================

def get_risk_level(probability):
    """
    Probability must be expressed as a percentage from 0 to 100.
    """
    if probability < 30:
        return "Low Risk"
    elif probability < 60:
        return "Medium Risk"
    else:
        return "High Risk"


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_prediction(result, probability, risk_level):
    new_record = pd.DataFrame([{
        "Prediction": result,
        "Probability": round(probability, 2),
        "Risk": risk_level
    }])

    if os.path.exists(HISTORY_FILE):
        old_history = pd.read_csv(HISTORY_FILE)
        history = pd.concat(
            [old_history, new_record],
            ignore_index=True
        )
    else:
        history = new_record

    history.to_csv(HISTORY_FILE, index=False)


# ============================================================
# LOAD PREDICTION HISTORY
# ============================================================

def get_history():
    if os.path.exists(HISTORY_FILE):
        history = pd.read_csv(HISTORY_FILE)

        if "Risk" not in history.columns:
            history["Risk"] = history["Probability"].apply(
                lambda value: get_risk_level(float(value))
            )
        else:
            history["Risk"] = history.apply(
                lambda row: (
                    row["Risk"]
                    if pd.notna(row["Risk"])
                    and str(row["Risk"]).strip()
                    else get_risk_level(float(row["Probability"]))
                ),
                axis=1
            )

        return history

    return pd.DataFrame(
        columns=["Prediction", "Probability", "Risk"]
    )


# ============================================================
# HOME AND CUSTOMER PREDICTION
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    probability = None
    risk_level = None
    error = None

    form_data = (
        request.form.to_dict()
        if request.method == "POST"
        else {}
    )

    if request.method == "POST":
        try:
            data = {
                "SeniorCitizen": int(request.form["SeniorCitizen"]),
                "tenure": float(request.form["tenure"]),
                "MonthlyCharges": float(
                    request.form["MonthlyCharges"]
                ),
                "TotalCharges": float(
                    request.form["TotalCharges"]
                ),
                "gender": request.form["gender"],
                "Partner": request.form["Partner"],
                "Dependents": request.form["Dependents"],
                "PhoneService": request.form["PhoneService"],
                "MultipleLines": request.form["MultipleLines"],
                "InternetService": request.form["InternetService"],
                "OnlineSecurity": request.form["OnlineSecurity"],
                "OnlineBackup": request.form["OnlineBackup"],
                "DeviceProtection": request.form["DeviceProtection"],
                "TechSupport": request.form["TechSupport"],
                "StreamingTV": request.form["StreamingTV"],
                "StreamingMovies": request.form["StreamingMovies"],
                "Contract": request.form["Contract"],
                "PaperlessBilling": request.form["PaperlessBilling"],
                "PaymentMethod": request.form["PaymentMethod"]
            }

            input_df = pd.DataFrame([data], columns=features)

            prediction = int(model.predict(input_df)[0])

            probability = float(
                model.predict_proba(input_df)[0][1]
            ) * 100

            if prediction == 1:
                result = "Customer Likely to Churn"
            else:
                result = "Customer Not Likely to Churn"

            risk_level = get_risk_level(probability)

            save_prediction(
                result,
                probability,
                risk_level
            )

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        result=result,
        probability=probability,
        risk_level=risk_level,
        error=error,
        form_data=form_data
    )


# ============================================================
# ANALYTICS DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():
    history = get_history()
    total_predictions = len(history)

    if total_predictions > 0:
        churn_count = len(
            history[
                history["Prediction"]
                == "Customer Likely to Churn"
            ]
        )

        safe_count = total_predictions - churn_count

        average_probability = round(
            history["Probability"].mean(), 2
        )

        churn_rate = round(
            (churn_count / total_predictions) * 100, 2
        )

        risk_counts = history["Probability"].apply(
            lambda value: get_risk_level(float(value))
        ).value_counts()

        low_risk_count = int(
            risk_counts.get("Low Risk", 0)
        )
        medium_risk_count = int(
            risk_counts.get("Medium Risk", 0)
        )
        high_risk_count = int(
            risk_counts.get("High Risk", 0)
        )

    else:
        churn_count = 0
        safe_count = 0
        average_probability = 0
        churn_rate = 0
        low_risk_count = 0
        medium_risk_count = 0
        high_risk_count = 0

    return render_template(
        "dashboard.html",
        total_predictions=total_predictions,
        churn_count=churn_count,
        safe_count=safe_count,
        average_probability=average_probability,
        churn_rate=churn_rate,
        low_risk_count=low_risk_count,
        medium_risk_count=medium_risk_count,
        high_risk_count=high_risk_count
    )


# ============================================================
# PREDICTION HISTORY
# ============================================================

@app.route("/history")
def history():
    history_data = get_history()
    records = history_data.to_dict(orient="records")

    return render_template(
        "history.html",
        records=records
    )


# ============================================================
# SETTINGS
# ============================================================

@app.route("/settings")
def settings():
    return render_template(
        "settings.html",
        model_name="Elastic Net Regularized Logistic Regression",
        feature_count=len(features),
        numerical_count=len(numerical_features),
        categorical_count=len(categorical_features)
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():
    return {
        "status": "ok",
        "model": "Elastic Net Logistic Regression"
    }


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
